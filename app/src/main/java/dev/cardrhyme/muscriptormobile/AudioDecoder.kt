package dev.cardrhyme.muscriptormobile

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.nio.ByteOrder
import kotlin.math.floor

class AudioDecoder(private val context: Context) {
    data class DecodedAudio(val samples16k: FloatArray, val durationSeconds: Double)

    suspend fun decode(uri: Uri, onProgress: (Float) -> Unit): DecodedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: error("No audio track found")
            extractor.selectTrack(trackIndex)
            val inputFormat = extractor.getTrackFormat(trackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME) ?: error("Audio MIME type missing")
            val durationUs = inputFormat.getLongOrDefault(MediaFormat.KEY_DURATION, 0L)
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(inputFormat, null, null, 0)
            codec.start()

            val output = FloatCollector()
            val info = MediaCodec.BufferInfo()
            var inputEnded = false
            var outputEnded = false
            var sampleRate = inputFormat.getIntOrDefault(MediaFormat.KEY_SAMPLE_RATE, 44_100)
            var channels = inputFormat.getIntOrDefault(MediaFormat.KEY_CHANNEL_COUNT, 2)
            var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT

            while (!outputEnded) {
                currentCoroutineContext().ensureActive()
                if (!inputEnded) {
                    val inputIndex = codec.dequeueInputBuffer(10_000)
                    if (inputIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputIndex) ?: error("Decoder input buffer missing")
                        inputBuffer.clear()
                        val size = extractor.readSampleData(inputBuffer, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                inputIndex,
                                0,
                                0,
                                0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputEnded = true
                        } else {
                            codec.queueInputBuffer(
                                inputIndex,
                                0,
                                size,
                                extractor.sampleTime.coerceAtLeast(0),
                                extractor.sampleFlags,
                            )
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = codec.dequeueOutputBuffer(info, 10_000)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val format = codec.outputFormat
                        sampleRate = format.getIntOrDefault(MediaFormat.KEY_SAMPLE_RATE, sampleRate)
                        channels = format.getIntOrDefault(MediaFormat.KEY_CHANNEL_COUNT, channels)
                        pcmEncoding = format.getIntOrDefault(
                            MediaFormat.KEY_PCM_ENCODING,
                            AudioFormat.ENCODING_PCM_16BIT,
                        )
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER, MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                    else -> if (outputIndex >= 0) {
                        val buffer = codec.getOutputBuffer(outputIndex)
                        if (buffer != null && info.size > 0) {
                            val view = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                            view.position(info.offset)
                            view.limit(info.offset + info.size)
                            appendPcm(view.slice().order(ByteOrder.LITTLE_ENDIAN), pcmEncoding, channels, output)
                        }
                        if (durationUs > 0) {
                            onProgress((info.presentationTimeUs.toDouble() / durationUs).toFloat().coerceIn(0f, 1f))
                        }
                        outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        codec.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }

            val mono = output.toArray()
            require(mono.isNotEmpty()) { "Decoder produced no PCM samples" }
            require(mono.size.toDouble() / sampleRate <= 30.0 * 60.0) {
                "Audio longer than 30 minutes is not supported yet"
            }
            val resampled = if (sampleRate == TARGET_SAMPLE_RATE) mono else linearResample(
                mono,
                sampleRate,
                TARGET_SAMPLE_RATE,
            )
            onProgress(1f)
            return DecodedAudio(resampled, resampled.size.toDouble() / TARGET_SAMPLE_RATE)
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            extractor.release()
        }
    }

    private fun appendPcm(
        buffer: java.nio.ByteBuffer,
        encoding: Int,
        channels: Int,
        output: FloatCollector,
    ) {
        require(channels > 0)
        when (encoding) {
            AudioFormat.ENCODING_PCM_FLOAT -> {
                val floats = buffer.asFloatBuffer()
                val frames = floats.remaining() / channels
                repeat(frames) {
                    var sum = 0f
                    repeat(channels) { sum += floats.get() }
                    output.add((sum / channels).coerceIn(-1f, 1f))
                }
            }
            AudioFormat.ENCODING_PCM_8BIT -> {
                val frames = buffer.remaining() / channels
                repeat(frames) {
                    var sum = 0f
                    repeat(channels) { sum += ((buffer.get().toInt() and 0xff) - 128) / 128f }
                    output.add(sum / channels)
                }
            }
            else -> {
                val shorts = buffer.asShortBuffer()
                val frames = shorts.remaining() / channels
                repeat(frames) {
                    var sum = 0f
                    repeat(channels) { sum += shorts.get() / 32768f }
                    output.add((sum / channels).coerceIn(-1f, 1f))
                }
            }
        }
    }

    private fun linearResample(input: FloatArray, inputRate: Int, outputRate: Int): FloatArray {
        val outputLength = floor(input.size.toDouble() * outputRate / inputRate).toInt().coerceAtLeast(1)
        val output = FloatArray(outputLength)
        val step = inputRate.toDouble() / outputRate
        for (index in output.indices) {
            val source = index * step
            val left = source.toInt().coerceAtMost(input.lastIndex)
            val right = (left + 1).coerceAtMost(input.lastIndex)
            val fraction = (source - left).toFloat()
            output[index] = input[left] + (input[right] - input[left]) * fraction
        }
        return output
    }

    private fun MediaFormat.getIntOrDefault(key: String, fallback: Int): Int =
        if (containsKey(key)) getInteger(key) else fallback

    private fun MediaFormat.getLongOrDefault(key: String, fallback: Long): Long =
        if (containsKey(key)) getLong(key) else fallback

    private class FloatCollector(initialCapacity: Int = 1 shl 20) {
        private var data = FloatArray(initialCapacity)
        private var size = 0

        fun add(value: Float) {
            if (size == data.size) data = data.copyOf(data.size * 2)
            data[size++] = value
        }

        fun toArray(): FloatArray = data.copyOf(size)
    }

    companion object {
        const val TARGET_SAMPLE_RATE = 16_000
    }
}
