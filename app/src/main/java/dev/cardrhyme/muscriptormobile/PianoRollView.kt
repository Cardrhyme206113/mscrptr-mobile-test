package dev.cardrhyme.muscriptormobile

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
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

    /**
     * Inference may emit many note events in a burst. Queue them and drain once per UI frame rather
     * than invalidating and redrawing the whole piano roll for every individual event.
     */
    private val pendingEvents = ConcurrentLinkedQueue<LiveNoteEvent>()
    private val drainPosted = AtomicBoolean(false)

    /** Completed notes are bucketed by onset second, so a 20-second viewport never scans the song. */
    private val completedBuckets = HashMap<Int, MutableList<MidiNote>>()
    private val longNotes = ArrayList<MidiNote>()
    private val active = LinkedHashMap<Long, ActiveNote>()

    private var cursorSeconds = 0.0
    private val minimumPitch = 24
    private val maximumPitch = 108
    private var darkMode = true

    private val backgroundPaint = Paint()
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = resources.displayMetrics.density
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 11f * resources.displayMetrics.scaledDensity
    }
    private val notePaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val cursorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 2f * resources.displayMetrics.density
    }
    private val programColors = IntArray(129)
    private val noteNames = arrayOf(
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    )

    init {
        updatePalette()
        clipToOutline = true
    }

    fun setDarkMode(enabled: Boolean) {
        if (darkMode == enabled) return
        darkMode = enabled
        updatePalette()
        invalidate()
    }

    private fun updatePalette() {
        if (darkMode) {
            backgroundPaint.color = Color.rgb(12, 14, 20)
            gridPaint.color = Color.argb(70, 210, 220, 255)
            labelPaint.color = Color.argb(190, 225, 230, 245)
            cursorPaint.color = Color.WHITE
        } else {
            backgroundPaint.color = Color.rgb(244, 245, 249)
            gridPaint.color = Color.argb(42, 25, 29, 42)
            labelPaint.color = Color.rgb(91, 97, 113)
            cursorPaint.color = Color.rgb(23, 24, 32)
        }
        for (program in programColors.indices) {
            val hue = if (program == 128) 8f else ((program * 47) % 360).toFloat()
            programColors[program] = Color.HSVToColor(
                floatArrayOf(
                    hue,
                    if (darkMode) 0.62f else 0.70f,
                    if (darkMode) 0.95f else 0.80f,
                ),
            )
        }
    }

    fun reset() {
        pendingEvents.clear()
        completedBuckets.clear()
        longNotes.clear()
        active.clear()
        cursorSeconds = 0.0
        invalidate()
    }

    /** Safe to call from the inference thread; events are applied on the UI thread in one batch. */
    fun accept(event: LiveNoteEvent) {
        pendingEvents.offer(event)
        scheduleDrain()
    }

    fun setCursor(seconds: Double) {
        val safe = seconds.coerceAtLeast(0.0)
        if (abs(safe - cursorSeconds) < CURSOR_REDRAW_THRESHOLD_SECONDS) return
        cursorSeconds = safe
        postInvalidateOnAnimation()
    }

    private fun scheduleDrain() {
        if (!drainPosted.compareAndSet(false, true)) return
        post {
            drainPosted.set(false)
            var changed = false
            while (true) {
                val event = pendingEvents.poll() ?: break
                applyEvent(event)
                changed = true
            }
            if (changed) postInvalidateOnAnimation()
            if (pendingEvents.isNotEmpty()) scheduleDrain()
        }
    }

    private fun applyEvent(event: LiveNoteEvent) {
        when (event) {
            is LiveNoteEvent.Started -> {
                active[event.id] = ActiveNote(
                    event.id,
                    event.program,
                    event.pitch,
                    event.onsetSeconds,
                    event.isDrum,
                )
            }
            is LiveNoteEvent.Ended -> {
                active.remove(event.id)
                val note = event.note
                if (note.offsetSeconds - note.onsetSeconds > WINDOW_SECONDS) {
                    longNotes += note
                } else {
                    completedBuckets.getOrPut(floor(note.onsetSeconds).toInt()) {
                        ArrayList()
                    } += note
                }
            }
        }
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
        val visibleEnd = max(WINDOW_SECONDS, cursorSeconds + 1.0)
        val visibleStart = max(0.0, visibleEnd - WINDOW_SECONDS)

        for (pitch in minimumPitch..maximumPitch step 12) {
            val y = yForPitch(pitch, top, plotHeight)
            canvas.drawLine(labelWidth, y, width.toFloat(), y, gridPaint)
            canvas.drawText(noteName(pitch), 4f * density, y + 4f * density, labelPaint)
        }

        val firstSecond = floor(visibleStart).toInt()
        val lastSecond = ceil(visibleEnd).toInt()
        for (second in firstSecond..lastSecond) {
            val x = xForTime(second.toDouble(), visibleStart, labelWidth, plotWidth)
            canvas.drawLine(x, top, x, bottom, gridPaint)
            if (second % 5 == 0) {
                canvas.drawText("${second}s", x + 2f, height - 4f, labelPaint)
            }
        }

        val firstBucket = floor(visibleStart - WINDOW_SECONDS).toInt()
        for (bucket in firstBucket..lastSecond) {
            completedBuckets[bucket]?.forEach { note ->
                if (note.offsetSeconds >= visibleStart && note.onsetSeconds <= visibleEnd) {
                    drawNote(
                        canvas,
                        note.program,
                        note.pitch,
                        note.onsetSeconds,
                        note.offsetSeconds,
                        note.isDrum,
                        visibleStart,
                        labelWidth,
                        plotWidth,
                        top,
                        plotHeight,
                    )
                }
            }
        }

        longNotes.forEach { note ->
            if (note.offsetSeconds >= visibleStart && note.onsetSeconds <= visibleEnd) {
                drawNote(
                    canvas,
                    note.program,
                    note.pitch,
                    note.onsetSeconds,
                    note.offsetSeconds,
                    note.isDrum,
                    visibleStart,
                    labelWidth,
                    plotWidth,
                    top,
                    plotHeight,
                )
            }
        }

        active.values.forEach { note ->
            if (note.onset <= visibleEnd) {
                drawNote(
                    canvas,
                    note.program,
                    note.pitch,
                    note.onset,
                    max(note.onset + 0.04, cursorSeconds),
                    note.isDrum,
                    visibleStart,
                    labelWidth,
                    plotWidth,
                    top,
                    plotHeight,
                )
            }
        }

        val cursorX = xForTime(cursorSeconds, visibleStart, labelWidth, plotWidth)
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
        labelWidth: Float,
        plotWidth: Float,
        top: Float,
        plotHeight: Float,
    ) {
        val x1 = xForTime(onset, visibleStart, labelWidth, plotWidth)
        val x2 = xForTime(
            max(offset, onset + if (isDrum) 0.08 else 0.03),
            visibleStart,
            labelWidth,
            plotWidth,
        )
        val noteHeight = plotHeight / (maximumPitch - minimumPitch + 1)
        val y = yForPitch(pitch, top, plotHeight)
        notePaint.color = programColors[program.coerceIn(0, 128)]
        canvas.drawRoundRect(
            x1,
            y - noteHeight * 0.85f,
            max(x1 + 2f, x2),
            y,
            2f,
            2f,
            notePaint,
        )
    }

    private fun xForTime(
        time: Double,
        visibleStart: Double,
        labelWidth: Float,
        plotWidth: Float,
    ): Float = labelWidth + (((time - visibleStart) / WINDOW_SECONDS) * plotWidth).toFloat()

    private fun yForPitch(pitch: Int, top: Float, plotHeight: Float): Float {
        val normalized = (pitch.coerceIn(minimumPitch, maximumPitch) - minimumPitch).toFloat() /
            (maximumPitch - minimumPitch)
        return top + plotHeight * (1f - normalized)
    }

    private fun noteName(pitch: Int): String =
        "${noteNames[pitch % 12]}${pitch / 12 - 1}"

    companion object {
        private const val WINDOW_SECONDS = 20.0
        private const val CURSOR_REDRAW_THRESHOLD_SECONDS = 0.025
    }
}
