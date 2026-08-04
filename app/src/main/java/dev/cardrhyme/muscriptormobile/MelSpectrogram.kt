package dev.cardrhyme.muscriptormobile

import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

object MelSpectrogram {
    const val SAMPLE_RATE = 16_000
    const val SEGMENT_SAMPLES = 80_000
    const val N_FFT = 2_048
    const val HOP_LENGTH = 160
    const val N_MELS = 512
    const val FRAMES = 501

    private val fft = FloatFFT_1D(N_FFT.toLong())
    private val window = FloatArray(N_FFT) { index ->
        (0.5 - 0.5 * cos(2.0 * PI * index / N_FFT)).toFloat()
    }
    private val filters: Array<SparseFilter> = buildFilters()

    @Synchronized
    fun compute(source: FloatArray, offset: Int): FloatArray {
        val audio = FloatArray(SEGMENT_SAMPLES)
        val available = (source.size - offset).coerceIn(0, SEGMENT_SAMPLES)
        if (available > 0) source.copyInto(audio, 0, offset, offset + available)

        val pad = N_FFT / 2
        val padded = FloatArray(SEGMENT_SAMPLES + N_FFT)
        audio.copyInto(padded, pad)
        for (index in 0 until pad) {
            padded[pad - 1 - index] = audio[index + 1]
            padded[pad + SEGMENT_SAMPLES + index] = audio[SEGMENT_SAMPLES - 2 - index]
        }

        val frame = FloatArray(N_FFT)
        val magnitude = FloatArray(N_FFT / 2 + 1)
        val result = FloatArray(FRAMES * N_MELS)

        for (frameIndex in 0 until FRAMES) {
            val start = frameIndex * HOP_LENGTH
            for (i in 0 until N_FFT) frame[i] = padded[start + i] * window[i]
            fft.realForward(frame)
            magnitude[0] = kotlin.math.abs(frame[0])
            magnitude[N_FFT / 2] = kotlin.math.abs(frame[1])
            for (bin in 1 until N_FFT / 2) {
                val real = frame[2 * bin]
                val imaginary = frame[2 * bin + 1]
                magnitude[bin] = sqrt(real * real + imaginary * imaginary)
            }

            val row = frameIndex * N_MELS
            for (melIndex in 0 until N_MELS) {
                val filter = filters[melIndex]
                var sum = 0f
                for (i in filter.bins.indices) sum += magnitude[filter.bins[i]] * filter.weights[i]
                result[row + melIndex] = ln(sum + 1e-6f)
            }
        }
        return result
    }

    private fun buildFilters(): Array<SparseFilter> {
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(SAMPLE_RATE / 2.0)
        val points = DoubleArray(N_MELS + 2) { index ->
            melToHz(melMin + (melMax - melMin) * index / (N_MELS + 1))
        }
        return Array(N_MELS) { mel ->
            val left = points[mel]
            val center = points[mel + 1]
            val right = points[mel + 2]
            val bins = ArrayList<Int>()
            val weights = ArrayList<Float>()
            for (bin in 0..N_FFT / 2) {
                val frequency = bin.toDouble() * SAMPLE_RATE / N_FFT
                val down = (frequency - left) / (center - left)
                val up = (right - frequency) / (right - center)
                val weight = max(0.0, min(down, up))
                if (weight > 0.0) {
                    bins += bin
                    weights += weight.toFloat()
                }
            }
            SparseFilter(bins.toIntArray(), weights.toFloatArray())
        }
    }

    private fun hzToMel(frequency: Double): Double = 2595.0 * log10(1.0 + frequency / 700.0)
    private fun melToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private data class SparseFilter(val bins: IntArray, val weights: FloatArray)
}
