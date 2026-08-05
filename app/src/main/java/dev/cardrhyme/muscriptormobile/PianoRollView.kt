package dev.cardrhyme.muscriptormobile

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
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
     * Inference can emit note events faster than the screen needs to redraw. Events are drained in
     * bounded batches and visual invalidation is capped, preventing the viewer from competing with
     * ONNX inference at 60-120 redraws per second.
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
    private var lastEventRedrawMs = 0L

    private val density = resources.displayMetrics.density
    private val backgroundPaint = Paint().apply { style = Paint.Style.FILL }
    private val gridPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = density
        isAntiAlias = false
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 11f * resources.displayMetrics.scaledDensity
    }
    private val notePaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = false
    }
    private val cursorPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * density
        isAntiAlias = false
    }
    private val outlinePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = density
        isAntiAlias = false
    }
    private val programColors = IntArray(129)
    private val noteNames = arrayOf(
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    )

    init {
        updatePalette()
    }

    fun setDarkMode(enabled: Boolean) {
        darkMode = enabled
        updatePalette()
        invalidate()
    }

    private fun updatePalette() {
        backgroundPaint.color = context.getColor(R.color.app_surface)
        gridPaint.color = withAlpha(context.getColor(R.color.app_outline), if (darkMode) 210 else 180)
        labelPaint.color = context.getColor(R.color.app_on_surface_variant)
        cursorPaint.color = context.getColor(R.color.app_primary)
        outlinePaint.color = context.getColor(R.color.app_outline)

        for (program in programColors.indices) {
            val hue = if (program == 128) 8f else ((program * 47) % 360).toFloat()
            programColors[program] = Color.HSVToColor(
                floatArrayOf(
                    hue,
                    if (darkMode) 0.58f else 0.70f,
                    if (darkMode) 0.96f else 0.78f,
                ),
            )
        }
    }

    private fun withAlpha(color: Int, alpha: Int): Int =
        (color and 0x00ffffff) or (alpha.coerceIn(0, 255) shl 24)

    fun reset() {
        pendingEvents.clear()
        completedBuckets.clear()
        longNotes.clear()
        active.clear()
        cursorSeconds = 0.0
        lastEventRedrawMs = 0L
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
        val elapsed = SystemClock.uptimeMillis() - lastEventRedrawMs
        val delay = (EVENT_REDRAW_INTERVAL_MILLIS - elapsed).coerceAtLeast(0L)
        postDelayed({ drainEvents() }, delay)
    }

    private fun drainEvents() {
        drainPosted.set(false)
        var applied = 0
        while (applied < MAX_EVENTS_PER_DRAIN) {
            val event = pendingEvents.poll() ?: break
            applyEvent(event)
            applied += 1
        }

        if (applied > 0) {
            lastEventRedrawMs = SystemClock.uptimeMillis()
            postInvalidateOnAnimation()
        }
        if (pendingEvents.isNotEmpty()) scheduleDrain()
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
        if (width <= 0 || height <= 0) return
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), backgroundPaint)

        val labelWidth = 38f * density
        val top = 8f * density
        val bottom = height - 18f * density
        val plotWidth = (width - labelWidth).coerceAtLeast(1f)
        val plotHeight = (bottom - top).coerceAtLeast(1f)
        val visibleEnd = max(WINDOW_SECONDS, cursorSeconds + 1.0)
        val visibleStart = max(0.0, visibleEnd - WINDOW_SECONDS)
        val pixelsPerSecond = plotWidth / WINDOW_SECONDS.toFloat()
        val pitchScale = plotHeight / (maximumPitch - minimumPitch).toFloat()
        val noteHeight = plotHeight / (maximumPitch - minimumPitch + 1)

        for (pitch in minimumPitch..maximumPitch step 12) {
            val y = yForPitch(pitch, top, pitchScale)
            canvas.drawLine(labelWidth, y, width.toFloat(), y, gridPaint)
            canvas.drawText(noteName(pitch), 4f * density, y + 4f * density, labelPaint)
        }

        val firstSecond = floor(visibleStart).toInt()
        val lastSecond = ceil(visibleEnd).toInt()
        for (second in firstSecond..lastSecond) {
            val x = xForTime(second.toDouble(), visibleStart, labelWidth, pixelsPerSecond)
            canvas.drawLine(x, top, x, bottom, gridPaint)
            if (second % 5 == 0) {
                canvas.drawText("${second}s", x + 2f, height - 4f, labelPaint)
            }
        }

        val saveCount = canvas.save()
        canvas.clipRect(labelWidth, top, width.toFloat(), bottom)

        val firstBucket = floor(visibleStart - WINDOW_SECONDS).toInt()
        for (bucket in firstBucket..lastSecond) {
            val notes = completedBuckets[bucket] ?: continue
            for (index in notes.indices) {
                val note = notes[index]
                if (note.offsetSeconds >= visibleStart && note.onsetSeconds <= visibleEnd) {
                    drawNote(
                        canvas = canvas,
                        program = note.program,
                        pitch = note.pitch,
                        onset = note.onsetSeconds,
                        offset = note.offsetSeconds,
                        isDrum = note.isDrum,
                        visibleStart = visibleStart,
                        labelWidth = labelWidth,
                        pixelsPerSecond = pixelsPerSecond,
                        top = top,
                        pitchScale = pitchScale,
                        noteHeight = noteHeight,
                    )
                }
            }
        }

        for (index in longNotes.indices) {
            val note = longNotes[index]
            if (note.offsetSeconds >= visibleStart && note.onsetSeconds <= visibleEnd) {
                drawNote(
                    canvas = canvas,
                    program = note.program,
                    pitch = note.pitch,
                    onset = note.onsetSeconds,
                    offset = note.offsetSeconds,
                    isDrum = note.isDrum,
                    visibleStart = visibleStart,
                    labelWidth = labelWidth,
                    pixelsPerSecond = pixelsPerSecond,
                    top = top,
                    pitchScale = pitchScale,
                    noteHeight = noteHeight,
                )
            }
        }

        for (note in active.values) {
            if (note.onset <= visibleEnd) {
                drawNote(
                    canvas = canvas,
                    program = note.program,
                    pitch = note.pitch,
                    onset = note.onset,
                    offset = max(note.onset + 0.04, cursorSeconds),
                    isDrum = note.isDrum,
                    visibleStart = visibleStart,
                    labelWidth = labelWidth,
                    pixelsPerSecond = pixelsPerSecond,
                    top = top,
                    pitchScale = pitchScale,
                    noteHeight = noteHeight,
                )
            }
        }

        val cursorX = xForTime(cursorSeconds, visibleStart, labelWidth, pixelsPerSecond)
        canvas.drawLine(cursorX, top, cursorX, bottom, cursorPaint)
        canvas.restoreToCount(saveCount)

        canvas.drawRect(
            outlinePaint.strokeWidth / 2f,
            outlinePaint.strokeWidth / 2f,
            width - outlinePaint.strokeWidth / 2f,
            height - outlinePaint.strokeWidth / 2f,
            outlinePaint,
        )
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
        pixelsPerSecond: Float,
        top: Float,
        pitchScale: Float,
        noteHeight: Float,
    ) {
        val x1 = xForTime(onset, visibleStart, labelWidth, pixelsPerSecond)
        val x2 = xForTime(
            max(offset, onset + if (isDrum) 0.08 else 0.03),
            visibleStart,
            labelWidth,
            pixelsPerSecond,
        )
        val y = yForPitch(pitch, top, pitchScale)
        notePaint.color = programColors[program.coerceIn(0, 128)]
        canvas.drawRect(
            x1,
            y - noteHeight * 0.82f,
            max(x1 + 2f, x2),
            y,
            notePaint,
        )
    }

    private fun xForTime(
        time: Double,
        visibleStart: Double,
        labelWidth: Float,
        pixelsPerSecond: Float,
    ): Float = labelWidth + ((time - visibleStart) * pixelsPerSecond).toFloat()

    private fun yForPitch(pitch: Int, top: Float, pitchScale: Float): Float =
        top + (maximumPitch - pitch.coerceIn(minimumPitch, maximumPitch)) * pitchScale

    private fun noteName(pitch: Int): String =
        "${noteNames[pitch % 12]}${pitch / 12 - 1}"

    companion object {
        private const val WINDOW_SECONDS = 20.0
        private const val CURSOR_REDRAW_THRESHOLD_SECONDS = 0.05
        private const val EVENT_REDRAW_INTERVAL_MILLIS = 50L
        private const val MAX_EVENTS_PER_DRAIN = 512
    }
}
