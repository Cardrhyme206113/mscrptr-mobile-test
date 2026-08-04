package dev.cardrhyme.muscriptormobile

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class PianoRollView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private data class ActiveNote(
        val id: Long,
        val program: Int,
        val pitch: Int,
        val onset: Double,
        val isDrum: Boolean,
    )

    private val completed = ArrayList<MidiNote>()
    private val active = LinkedHashMap<Long, ActiveNote>()
    private var cursorSeconds = 0.0
    private var minimumPitch = 24
    private var maximumPitch = 108

    private val backgroundPaint = Paint().apply { color = Color.rgb(12, 14, 20) }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(70, 210, 220, 255)
        strokeWidth = resources.displayMetrics.density
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(190, 225, 230, 245)
        textSize = 11f * resources.displayMetrics.scaledDensity
    }
    private val notePaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val cursorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 2f * resources.displayMetrics.density
    }

    fun reset() {
        completed.clear()
        active.clear()
        cursorSeconds = 0.0
        invalidate()
    }

    fun accept(event: LiveNoteEvent) {
        when (event) {
            is LiveNoteEvent.Started -> {
                active[event.id] = ActiveNote(
                    event.id,
                    event.program,
                    event.pitch,
                    event.onsetSeconds,
                    event.isDrum,
                )
                cursorSeconds = max(cursorSeconds, event.onsetSeconds)
            }
            is LiveNoteEvent.Ended -> {
                active.remove(event.id)
                completed += event.note
                cursorSeconds = max(cursorSeconds, event.note.offsetSeconds)
            }
        }
        invalidate()
    }

    fun setCursor(seconds: Double) {
        cursorSeconds = max(cursorSeconds, seconds)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), backgroundPaint)
        if (width <= 0 || height <= 0) return

        val density = resources.displayMetrics.density
        val labelWidth = 38f * density
        val top = 8f * density
        val bottom = height - 18f * density
        val plotWidth = width - labelWidth
        val plotHeight = (bottom - top).coerceAtLeast(1f)
        val windowSeconds = 20.0
        val visibleEnd = max(windowSeconds, cursorSeconds + 1.0)
        val visibleStart = max(0.0, visibleEnd - windowSeconds)

        for (pitch in minimumPitch..maximumPitch step 12) {
            val y = yForPitch(pitch, top, plotHeight)
            canvas.drawLine(labelWidth, y, width.toFloat(), y, gridPaint)
            canvas.drawText(noteName(pitch), 4f * density, y + 4f * density, labelPaint)
        }
        val firstSecond = visibleStart.toInt()
        val lastSecond = visibleEnd.toInt() + 1
        for (second in firstSecond..lastSecond) {
            val x = xForTime(second.toDouble(), visibleStart, windowSeconds, labelWidth, plotWidth)
            canvas.drawLine(x, top, x, bottom, gridPaint)
            if (second % 5 == 0) canvas.drawText("${second}s", x + 2f, height - 4f, labelPaint)
        }

        for (note in completed) {
            if (note.offsetSeconds < visibleStart || note.onsetSeconds > visibleEnd) continue
            drawNote(
                canvas,
                note.program,
                note.pitch,
                note.onsetSeconds,
                note.offsetSeconds,
                note.isDrum,
                visibleStart,
                windowSeconds,
                labelWidth,
                plotWidth,
                top,
                plotHeight,
            )
        }
        for (note in active.values) {
            if (note.onset > visibleEnd) continue
            drawNote(
                canvas,
                note.program,
                note.pitch,
                note.onset,
                max(note.onset + 0.04, cursorSeconds),
                note.isDrum,
                visibleStart,
                windowSeconds,
                labelWidth,
                plotWidth,
                top,
                plotHeight,
            )
        }

        val cursorX = xForTime(cursorSeconds, visibleStart, windowSeconds, labelWidth, plotWidth)
        canvas.drawLine(cursorX, top, cursorX, bottom, cursorPaint)
    }

    private fun drawNote(
        canvas: Canvas,
        program: Int,
        pitch: Int,
        onset: Double,
        offset: Double,
        isDrum: Boolean,
        visibleStart: Double,
        windowSeconds: Double,
        labelWidth: Float,
        plotWidth: Float,
        top: Float,
        plotHeight: Float,
    ) {
        val x1 = xForTime(onset, visibleStart, windowSeconds, labelWidth, plotWidth)
        val x2 = xForTime(max(offset, onset + if (isDrum) 0.08 else 0.03), visibleStart, windowSeconds, labelWidth, plotWidth)
        val noteHeight = plotHeight / (maximumPitch - minimumPitch + 1)
        val y = yForPitch(pitch, top, plotHeight)
        notePaint.color = colorForProgram(program)
        canvas.drawRoundRect(
            RectF(x1, y - noteHeight * 0.85f, max(x1 + 2f, x2), y),
            2f,
            2f,
            notePaint,
        )
    }

    private fun xForTime(
        time: Double,
        visibleStart: Double,
        windowSeconds: Double,
        labelWidth: Float,
        plotWidth: Float,
    ): Float = labelWidth + (((time - visibleStart) / windowSeconds) * plotWidth).toFloat()

    private fun yForPitch(pitch: Int, top: Float, plotHeight: Float): Float {
        val normalized = (pitch.coerceIn(minimumPitch, maximumPitch) - minimumPitch).toFloat() /
            (maximumPitch - minimumPitch)
        return top + plotHeight * (1f - normalized)
    }

    private fun colorForProgram(program: Int): Int {
        val hue = if (program == 128) 8f else ((program * 47) % 360).toFloat()
        return Color.HSVToColor(floatArrayOf(hue, 0.62f, 0.95f))
    }

    private fun noteName(pitch: Int): String {
        val names = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        return "${names[pitch % 12]}${pitch / 12 - 1}"
    }
}
