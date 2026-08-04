package dev.cardrhyme.muscriptormobile

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Process
import java.io.Closeable
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/**
 * Lightweight preview synth designed to coexist with ONNX inference.
 *
 * The previous renderer evaluated several sin/asin/exp functions for every sample of every voice,
 * held a lock while doing it, ran at maximum Java priority, and rendered continuously even when
 * silent. On the target phone that could steal roughly half of the decoder's throughput.
 *
 * This version uses a precomputed wavetable, stateful envelopes, mono PCM16 output, a bounded
 * polyphony limit, and a lock-free command queue. The audio thread owns all voice state, so note
 * scheduling never waits for an audio buffer to finish rendering.
 */
class LiveMidiSynth : Closeable {
    private sealed interface Command {
        data class On(
            val id: Long,
            val program: Int,
            val pitch: Int,
            val isDrum: Boolean,
        ) : Command

        data class Off(val id: Long) : Command
        data object AllOff : Command
    }

    private data class Voice(
        val id: Long,
        val program: Int,
        val pitch: Int,
        val isDrum: Boolean,
        val startedOrder: Long,
        val phaseStep: Int,
        val attackStep: Float,
        val sustainMultiplier: Float,
        val releaseMultiplier: Float,
        val drumMultiplier: Float,
        var phase: Int = 0,
        var level: Float = if (isDrum) 1f else 0f,
        var released: Boolean = false,
        var ageFrames: Long = 0,
        var noiseState: Int = (id xor (pitch.toLong() shl 16)).toInt().let {
            if (it == 0) 0x13579BDF else it
        },
    )

    private val commands = ConcurrentLinkedQueue<Command>()
    private val voices = LinkedHashMap<Long, Voice>()
    private var voiceOrder = 0L

    @Volatile
    private var running = true

    @Volatile
    private var masterVolume = 0.8f

    private val output = ShortArray(FRAMES_PER_BUFFER)
    private val sineTable = FloatArray(TABLE_SIZE) { index ->
        sin(2.0 * PI * index / TABLE_SIZE).toFloat()
    }

    private val audioTrack: AudioTrack
    private val renderThread: Thread
    private val configuredBufferFrames: Int

    /** Approximate queued hardware/software output latency used by the song/MIDI scheduler. */
    val outputLatencySeconds: Double
        get() = (configuredBufferFrames + FRAMES_PER_BUFFER * 0.5) / SAMPLE_RATE.toDouble()

    init {
        val minBytes = AudioTrack.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        check(minBytes > 0) { "MIDI AudioTrack configuration is unsupported: $minBytes" }

        val bufferBytes = maxOf(
            minBytes,
            FRAMES_PER_BUFFER * Short.SIZE_BYTES * 2,
        )
        configuredBufferFrames = bufferBytes / Short.SIZE_BYTES

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(bufferBytes)
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .build()

        check(audioTrack.state == AudioTrack.STATE_INITIALIZED) {
            "MIDI AudioTrack failed to initialize"
        }
        audioTrack.setVolume(1f)
        audioTrack.play()

        renderThread = Thread(::renderLoop, "MuScriptor-midi-synth").apply {
            start()
        }
    }

    fun setVolume(value: Float) {
        masterVolume = value.coerceIn(0f, 1f)
    }

    fun noteOn(id: Long, program: Int, pitch: Int, isDrum: Boolean) {
        commands.offer(
            Command.On(
                id = id,
                program = program.coerceIn(0, 128),
                pitch = pitch.coerceIn(0, 127),
                isDrum = isDrum,
            ),
        )
    }

    fun noteOff(id: Long) {
        commands.offer(Command.Off(id))
    }

    fun allNotesOff() {
        commands.offer(Command.AllOff)
    }

    private fun renderLoop() {
        // Audio priority prevents underruns, but the renderer is intentionally cheap enough not to
        // compete meaningfully with ONNX's worker pool.
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)

        while (running) {
            drainCommands()
            output.fill(0)

            if (voices.isNotEmpty()) {
                renderVoices()
            }

            val written = audioTrack.write(
                output,
                0,
                output.size,
                AudioTrack.WRITE_BLOCKING,
            )
            if (written < 0 && running) Thread.sleep(2)
        }
    }

    private fun drainCommands() {
        while (true) {
            when (val command = commands.poll() ?: break) {
                is Command.On -> {
                    val voice = createVoice(command)
                    voices[command.id] = voice
                    while (voices.size > MAX_POLYPHONY) {
                        val oldest = voices.values.minByOrNull { it.startedOrder } ?: break
                        voices.remove(oldest.id)
                    }
                }
                is Command.Off -> {
                    val voice = voices[command.id] ?: continue
                    // Percussion uses its own short natural decay. A 10 ms model drum note can have
                    // note-on and note-off land in the same scheduler tick; releasing it here would
                    // suppress the hit entirely.
                    if (!voice.isDrum) voice.released = true
                }
                Command.AllOff -> voices.clear()
            }
        }
    }

    private fun createVoice(command: Command.On): Voice {
        val frequency = 440.0 * 2.0.pow((command.pitch - 69) / 12.0)
        val phaseStep = (frequency * PHASE_RANGE / SAMPLE_RATE).toLong().toInt()

        val attackSeconds = when (command.program) {
            in 40..55 -> 0.045
            in 88..95 -> 0.075
            else -> 0.006
        }
        val releaseSeconds = when (command.program) {
            in 40..55, in 88..95 -> 0.22
            else -> 0.085
        }
        val naturalDecayPerSecond = when (command.program) {
            in 0..15 -> 0.38
            in 24..39 -> 0.72
            else -> 0.0
        }
        val drumDecayPerSecond = when (command.pitch) {
            in 35..36 -> 8.0
            in 49..57 -> 4.0
            else -> 14.0
        }

        return Voice(
            id = command.id,
            program = command.program,
            pitch = command.pitch,
            isDrum = command.isDrum,
            startedOrder = voiceOrder++,
            phaseStep = phaseStep,
            attackStep = (1.0 / (attackSeconds * SAMPLE_RATE)).toFloat(),
            sustainMultiplier = exp(-naturalDecayPerSecond / SAMPLE_RATE).toFloat(),
            releaseMultiplier = exp(
                ln(SILENCE_THRESHOLD.toDouble()) / (releaseSeconds * SAMPLE_RATE),
            ).toFloat(),
            drumMultiplier = exp(-drumDecayPerSecond / SAMPLE_RATE).toFloat(),
        )
    }

    private fun renderVoices() {
        val volume = masterVolume
        val iterator = voices.values.iterator()

        while (iterator.hasNext()) {
            val voice = iterator.next()
            var dead = false

            for (frame in 0 until FRAMES_PER_BUFFER) {
                updateEnvelope(voice)
                if (voice.level < SILENCE_THRESHOLD && (voice.released || voice.isDrum)) {
                    dead = true
                    break
                }

                val sample = waveform(voice) * voice.level * VOICE_GAIN
                val mixed = output[frame] / 32768f + sample * volume
                output[frame] = (mixed.coerceIn(-1f, 1f) * 32767f).toInt().toShort()

                voice.phase = voice.phase + voice.phaseStep
                voice.ageFrames += 1
            }

            if (dead || voice.ageFrames > MAX_VOICE_SECONDS * SAMPLE_RATE.toLong()) {
                iterator.remove()
            }
        }
    }

    private fun updateEnvelope(voice: Voice) {
        if (voice.isDrum) {
            voice.level *= voice.drumMultiplier
            return
        }

        if (voice.released) {
            voice.level *= voice.releaseMultiplier
            return
        }

        if (voice.level < 1f) {
            voice.level = min(1f, voice.level + voice.attackStep)
        }
        voice.level *= voice.sustainMultiplier
    }

    private fun waveform(voice: Voice): Float {
        val index = voice.phase ushr PHASE_SHIFT

        if (voice.isDrum) {
            val noise = nextNoise(voice)
            return when (voice.pitch) {
                in 35..36 -> sineTable[index] * 0.82f + noise * 0.18f
                in 38..40 -> noise * 0.90f
                in 42..46 -> noise * 0.62f
                else -> noise * 0.74f + sineTable[index] * 0.26f
            }
        }

        val sine = sineTable[index]
        val saw = index * TABLE_TO_BIPOLAR - 1f
        val triangle = 1f - 4f * abs(index * TABLE_TO_UNIT - 0.5f)
        val square = if (index < TABLE_SIZE / 2) 1f else -1f

        return when (voice.program) {
            in 0..7 -> sine * 0.76f + sineTable[(index * 2) and TABLE_MASK] * 0.24f
            in 8..15 -> sine * 0.66f + triangle * 0.34f
            in 16..23 -> square * 0.46f + sine * 0.54f
            in 24..31 -> triangle * 0.68f + saw * 0.32f
            in 32..39 -> sine * 0.70f + square * 0.30f
            in 40..55 -> saw * 0.56f + triangle * 0.44f
            in 56..63 -> saw * 0.62f + square * 0.38f
            in 64..79 -> sine * 0.82f + sineTable[(index * 2) and TABLE_MASK] * 0.18f
            in 80..87 -> saw * 0.50f + square * 0.50f
            in 88..95 -> triangle * 0.64f + sine * 0.36f
            else -> sine
        }
    }

    private fun nextNoise(voice: Voice): Float {
        var value = voice.noiseState
        value = value xor (value shl 13)
        value = value xor (value ushr 17)
        value = value xor (value shl 5)
        voice.noiseState = value
        return ((value ushr 8) and 0x00ffffff) / 8_388_608f - 1f
    }

    override fun close() {
        if (!running) return
        running = false
        commands.clear()
        runCatching { audioTrack.stop() }
        runCatching { renderThread.join(700) }
        voices.clear()
        runCatching { audioTrack.flush() }
        runCatching { audioTrack.release() }
    }

    companion object {
        private const val SAMPLE_RATE = 48_000
        private const val FRAMES_PER_BUFFER = 256
        private const val MAX_POLYPHONY = 48
        private const val MAX_VOICE_SECONDS = 20
        private const val SILENCE_THRESHOLD = 0.00025f
        private const val VOICE_GAIN = 0.22f

        private const val TABLE_BITS = 11
        private const val TABLE_SIZE = 1 shl TABLE_BITS
        private const val TABLE_MASK = TABLE_SIZE - 1
        private const val PHASE_SHIFT = 32 - TABLE_BITS
        private const val PHASE_RANGE = 4_294_967_296.0
        private const val TABLE_TO_BIPOLAR = 2f / (TABLE_SIZE - 1)
        private const val TABLE_TO_UNIT = 1f / (TABLE_SIZE - 1)
    }
}
