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

sealed interface LiveNoteEvent {
    data class Started(
        val id: Long,
        val program: Int,
        val pitch: Int,
        val onsetSeconds: Double,
        val isDrum: Boolean,
    ) : LiveNoteEvent

    data class Ended(val id: Long, val note: MidiNote) : LiveNoteEvent
}

class TokenDecoder(
    private val frameRate: Int = 100,
    private val onLiveEvent: (LiveNoteEvent) -> Unit = {},
) {
    private data class Key(val program: Int, val pitch: Int)
    private data class Active(val id: Long, val onset: Double)
    private enum class Type { SPECIAL, SHIFT, PITCH, VELOCITY, TIE, PROGRAM, DRUM }
    private data class Event(val type: Type, val value: Int)

    private val open = LinkedHashMap<Key, Active>()
    private val notes = ArrayList<MidiNote>()
    private var nextId = 0L
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
                if (nextSeekTime == null || time < nextSeekTime!!) emitDrum(event.value, time)
            }
            Type.PITCH -> {
                val currentProgram = program ?: return
                val currentVelocity = velocity ?: return
                val time = tickState.toDouble() / frameRate
                if (nextSeekTime != null && time >= nextSeekTime!!) return
                val key = Key(currentProgram, event.value)
                if (key in open) close(key, time)
                if (currentVelocity > 0) open[key] = start(key, time)
            }
            else -> Unit
        }
    }

    fun finish(audioDurationSeconds: Double): List<MidiNote> {
        if (chunkStarted && inPrologue) {
            closeAll(seekTime)
        } else {
            val remaining = open.keys.toList()
            remaining.forEach { key ->
                val active = open[key] ?: return@forEach
                close(
                    key,
                    max(
                        active.onset + MIN_NOTE_SECONDS,
                        audioDurationSeconds.coerceAtMost(active.onset + 10.0),
                    ),
                )
            }
        }
        return notes.sortedWith(compareBy<MidiNote> { it.onsetSeconds }.thenBy { it.program }.thenBy { it.pitch })
    }

    fun openKeys(): List<Pair<Int, Int>> = open.keys.map { it.program to it.pitch }.sortedWith(
        compareBy<Pair<Int, Int>> { it.first }.thenBy { it.second },
    )

    private fun start(key: Key, onset: Double): Active {
        val active = Active(nextId++, onset)
        onLiveEvent(
            LiveNoteEvent.Started(
                active.id,
                key.program,
                key.pitch,
                onset,
                false,
            ),
        )
        return active
    }

    private fun emitDrum(pitch: Int, time: Double) {
        val id = nextId++
        onLiveEvent(LiveNoteEvent.Started(id, 128, pitch, time, true))
        val note = MidiNote(128, pitch, time, time + MIN_NOTE_SECONDS, true)
        notes += note
        onLiveEvent(LiveNoteEvent.Ended(id, note))
    }

    private fun close(key: Key, requestedTime: Double) {
        val active = open.remove(key) ?: return
        val note = MidiNote(
            key.program,
            key.pitch,
            active.onset,
            max(requestedTime, active.onset + MIN_NOTE_SECONDS),
            false,
        )
        notes += note
        onLiveEvent(LiveNoteEvent.Ended(active.id, note))
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
