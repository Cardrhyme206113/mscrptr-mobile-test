package dev.cardrhyme.muscriptormobile

import kotlin.math.max
import kotlin.math.roundToInt

data class MidiNote(
    val program: Int,
    val pitch: Int,
    val onsetSeconds: Double,
    val offsetSeconds: Double,
    val isDrum: Boolean,
)

class TokenDecoder(private val frameRate: Int = 100) {
    private data class Key(val program: Int, val pitch: Int)
    private enum class Type { SPECIAL, SHIFT, PITCH, VELOCITY, TIE, PROGRAM, DRUM }
    private data class Event(val type: Type, val value: Int)

    private val open = LinkedHashMap<Key, Double>()
    private val notes = ArrayList<MidiNote>()
    private var seekTime = 0.0
    private var nextSeekTime: Double? = null
    private var startTick = 0
    private var tickState = 0
    private var program: Int? = null
    private var velocity: Int? = null
    private var inPrologue = true
    private var skipRest = false
    private var tieSet = HashSet<Key>()
    private var chunkStarted = false

    fun startChunk(seekSeconds: Double, nextSeekSeconds: Double?) {
        if (chunkStarted && inPrologue) closeAll(seekTime)
        seekTime = seekSeconds
        nextSeekTime = nextSeekSeconds
        startTick = (seekSeconds * frameRate).roundToInt()
        tickState = startTick
        program = null
        velocity = null
        inPrologue = true
        skipRest = false
        tieSet = HashSet()
        chunkStarted = true
    }

    fun feed(token: Int) {
        val event = decodeEvent(token) ?: return
        if (inPrologue) {
            when (event.type) {
                Type.TIE -> {
                    inPrologue = false
                    velocity = null
                    val ended = open.keys.filter { it !in tieSet }
                    ended.forEach { close(it, seekTime) }
                }
                Type.SHIFT -> {
                    inPrologue = false
                    skipRest = true
                    closeAll(seekTime)
                }
                Type.PROGRAM -> program = event.value
                Type.PITCH -> program?.let { tieSet += Key(it, event.value) }
                else -> Unit
            }
            return
        }
        if (skipRest) return

        when (event.type) {
            Type.SHIFT -> if (event.value > 0) tickState = startTick + event.value
            Type.PROGRAM -> program = event.value
            Type.VELOCITY -> velocity = event.value
            Type.DRUM -> {
                val time = tickState.toDouble() / frameRate
                if (nextSeekTime == null || time < nextSeekTime!!) {
                    notes += MidiNote(128, event.value, time, time + MIN_NOTE_SECONDS, true)
                }
            }
            Type.PITCH -> {
                val currentProgram = program ?: return
                val currentVelocity = velocity ?: return
                val time = tickState.toDouble() / frameRate
                if (nextSeekTime != null && time >= nextSeekTime!!) return
                val key = Key(currentProgram, event.value)
                if (key in open) close(key, time)
                if (currentVelocity > 0) open[key] = time
            }
            else -> Unit
        }
    }

    fun finish(audioDurationSeconds: Double): List<MidiNote> {
        if (chunkStarted && inPrologue) {
            closeAll(seekTime)
        } else {
            val remaining = open.toMap()
            open.clear()
            remaining.forEach { (key, onset) ->
                notes += MidiNote(
                    key.program,
                    key.pitch,
                    onset,
                    max(onset + MIN_NOTE_SECONDS, audioDurationSeconds.coerceAtMost(onset + 10.0)),
                    false,
                )
            }
        }
        return notes.sortedWith(compareBy<MidiNote> { it.onsetSeconds }.thenBy { it.program }.thenBy { it.pitch })
    }

    fun openKeys(): List<Pair<Int, Int>> = open.keys.map { it.program to it.pitch }.sortedWith(
        compareBy<Pair<Int, Int>> { it.first }.thenBy { it.second },
    )

    private fun close(key: Key, requestedTime: Double) {
        val onset = open.remove(key) ?: return
        notes += MidiNote(
            key.program,
            key.pitch,
            onset,
            max(requestedTime, onset + MIN_NOTE_SECONDS),
            false,
        )
    }

    private fun closeAll(time: Double) {
        val keys = open.keys.toList()
        keys.forEach { close(it, time) }
    }

    private fun decodeEvent(token: Int): Event? = when {
        token in 0..2 -> Event(Type.SPECIAL, token)
        token in 3..1003 -> Event(Type.SHIFT, token - 3)
        token in 1004..1131 -> Event(Type.PITCH, token - 1004)
        token in 1132..1133 -> Event(Type.VELOCITY, token - 1132)
        token == 1134 -> Event(Type.TIE, 0)
        token in 1135..1264 -> Event(Type.PROGRAM, token - 1135)
        token in 1265..1392 -> Event(Type.DRUM, token - 1265)
        else -> null
    }

    companion object {
        private const val MIN_NOTE_SECONDS = 0.01
    }
}
