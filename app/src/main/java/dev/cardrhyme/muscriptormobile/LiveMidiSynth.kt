package dev.cardrhyme.muscriptormobile

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.Closeable
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/**
 * Small dependency-free General-MIDI-ish preview synth.
 *
 * It is deliberately a preview instrument rather than a SoundFont renderer: the model's program,
 * pitch, onset and offset are audible immediately without adding another large asset to the APK.
 */
class LiveMidiSynth : Closeable {
    private data class Voice(
        val id: Long,
        val program: Int,
        val pitch: Int,
        val frequency: Double,
        val isDrum: Boolean,
        val startedOrder: Long,
        var phase: Double = 0.0,
        var ageSamples: Long = 0,
        var releaseSample: Long = -1,
        var noiseState: Int = (id xor (pitch.toLong() shl 16)).toInt().let { if (it == 0) 1 else it },
    )

    private val lock = Any()
    private val voices = LinkedHashMap<Long, Voice>()
    private var voiceOrder = 0L

    @Volatile
    private var running = true

    @Volatile
    private var masterVolume = 0.8f

    private val sampleRate = 48_000
    private val channelCount = 2
    private val framesPerBuffer = 512
    private val output = FloatArray(framesPerBuffer * channelCount)

    private val audioTrack: AudioTrack
    private val renderThread: Thread

    init {
        val minBytes = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_STEREO,
            AudioFormat.ENCODING_PCM_FLOAT,
        ).coerceAtLeast(output.size * Float.SIZE_BYTES)
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                    .build(),
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(minBytes * 2, output.size * Float.SIZE_BYTES * 4))
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .build()
        check(audioTrack.state == AudioTrack.STATE_INITIALIZED) {
            "MIDI AudioTrack failed to initialize"
        }
        audioTrack.setVolume(1f)
        audioTrack.play()
        renderThread = Thread(::renderLoop, "MuScriptor-midi-synth").apply {
            priority = Thread.MAX_PRIORITY
            start()
        }
    }

    fun setVolume(value: Float) {
        masterVolume = value.coerceIn(0f, 1f)
    }

    fun noteOn(id: Long, program: Int, pitch: Int, isDrum: Boolean) {
        val safePitch = pitch.coerceIn(0, 127)
        val frequency = 440.0 * 2.0.pow((safePitch - 69) / 12.0)
        synchronized(lock) {
            voices[id] = Voice(
                id = id,
                program = program.coerceIn(0, 127),
                pitch = safePitch,
                frequency = frequency,
                isDrum = isDrum,
                startedOrder = voiceOrder++,
            )
            while (voices.size > MAX_POLYPHONY) {
                val oldest = voices.values.minByOrNull { it.startedOrder } ?: break
                voices.remove(oldest.id)
            }
        }
    }

    fun noteOff(id: Long) {
        synchronized(lock) {
            val voice = voices[id] ?: return
            if (voice.releaseSample < 0) voice.releaseSample = voice.ageSamples
        }
    }

    fun allNotesOff() {
        synchronized(lock) {
            voices.clear()
        }
    }

    private fun renderLoop() {
        while (running) {
            output.fill(0f)
            synchronized(lock) {
                val iterator = voices.values.iterator()
                while (iterator.hasNext()) {
                    val voice = iterator.next()
                    var dead = false
                    for (frame in 0 until framesPerBuffer) {
                        val amplitude = envelope(voice)

                        // A normal melodic attack intentionally begins at amplitude 0. The old
                        // code treated that first zero sample as an already-finished voice and
                        // removed every note before it could become audible. Only released voices
                        // and naturally decaying drums are allowed to die at a low envelope level.
                        val releaseFinished = voice.releaseSample >= 0 && amplitude < SILENCE_THRESHOLD
                        val drumFinished = voice.isDrum && voice.ageSamples > 0 && amplitude < SILENCE_THRESHOLD
                        if (releaseFinished || drumFinished) {
                            dead = true
                            break
                        }

                        val sample = waveform(voice) * amplitude * 0.24f
                        val left = frame * 2
                        // Tiny program-dependent stereo spread keeps dense arrangements readable.
                        val pan = (((voice.program * 17 + voice.pitch) % 21) - 10) / 50f
                        output[left] += sample * (1f - pan)
                        output[left + 1] += sample * (1f + pan)
                        voice.phase += 2.0 * PI * voice.frequency / sampleRate
                        if (voice.phase >= 2.0 * PI) voice.phase -= 2.0 * PI
                        voice.ageSamples += 1
                    }
                    if (dead || voice.ageSamples > sampleRate * 20L) iterator.remove()
                }
            }

            val volume = masterVolume
            for (index in output.indices) {
                val value = output[index] * volume
                output[index] = value / (1f + abs(value))
            }
            val written = audioTrack.write(
                output,
                0,
                output.size,
                AudioTrack.WRITE_BLOCKING,
            )
            if (written < 0) Thread.sleep(4)
        }
    }

    private fun envelope(voice: Voice): Float {
        val age = voice.ageSamples.toDouble() / sampleRate
        if (voice.isDrum) {
            val decay = when (voice.pitch) {
                in 35..36 -> 7.0
                in 49..57 -> 2.8
                else -> 12.0
            }
            return exp(-age * decay).toFloat()
        }

        val attackSeconds = when (voice.program) {
            in 40..55 -> 0.10
            in 88..95 -> 0.18
            else -> 0.012
        }
        var gain = min(1.0, age / attackSeconds)
        gain *= when (voice.program) {
            in 0..7 -> 0.45 + 0.55 * exp(-age * 0.42)
            in 24..31 -> exp(-age * 0.55)
            in 8..15 -> exp(-age * 0.32)
            else -> 1.0
        }
        if (voice.releaseSample >= 0) {
            val releaseAge = (voice.ageSamples - voice.releaseSample).toDouble() / sampleRate
            val releaseSeconds = when (voice.program) {
                in 40..55, in 88..95 -> 0.35
                else -> 0.12
            }
            gain *= exp(-releaseAge * 7.0 / releaseSeconds)
        }
        return gain.toFloat()
    }

    private fun waveform(voice: Voice): Float {
        val phase = voice.phase
        if (voice.isDrum) {
            voice.noiseState = voice.noiseState * 1_664_525 + 1_013_904_223
            val noise = ((voice.noiseState ushr 8) and 0xffff) / 32768f - 1f
            return when (voice.pitch) {
                in 35..36 -> (sin(phase * 0.38) * 0.82 + noise * 0.18).toFloat()
                in 38..40 -> noise * 0.82f
                in 42..46 -> noise * 0.55f
                else -> (noise * 0.7 + sin(phase) * 0.3).toFloat()
            }
        }

        val sine = sin(phase)
        val triangle = 2.0 / PI * kotlin.math.asin(sine)
        val square = if (sine >= 0.0) 1.0 else -1.0
        val saw = phase / PI - 1.0
        return when (voice.program) {
            in 0..7 -> (sine + 0.34 * sin(phase * 2.0) + 0.13 * sin(phase * 3.0)).toFloat()
            in 8..15 -> (0.65 * sine + 0.35 * triangle).toFloat()
            in 16..23 -> (0.55 * square + 0.45 * sine).toFloat()
            in 24..31 -> (0.62 * triangle + 0.25 * saw + 0.13 * sin(phase * 2.0)).toFloat()
            in 32..39 -> (0.72 * sine + 0.28 * square).toFloat()
            in 40..55 -> (0.62 * saw + 0.38 * triangle).toFloat()
            in 56..63 -> (0.72 * saw + 0.28 * square).toFloat()
            in 64..79 -> (0.78 * sine + 0.22 * sin(phase * 2.0)).toFloat()
            in 80..87 -> (0.55 * saw + 0.45 * square).toFloat()
            in 88..95 -> (0.68 * triangle + 0.32 * sine).toFloat()
            else -> sine.toFloat()
        }
    }

    override fun close() {
        running = false
        runCatching { renderThread.join(800) }
        synchronized(lock) { voices.clear() }
        runCatching { audioTrack.pause() }
        runCatching { audioTrack.flush() }
        runCatching { audioTrack.release() }
    }

    companion object {
        private const val MAX_POLYPHONY = 96
        private const val SILENCE_THRESHOLD = 0.00025f
    }
}
