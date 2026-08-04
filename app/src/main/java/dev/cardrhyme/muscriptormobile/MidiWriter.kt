package dev.cardrhyme.muscriptormobile

import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt

object MidiWriter {
    private const val TICKS_PER_QUARTER = 480
    private const val MICROSECONDS_PER_QUARTER = 500_000
    private const val TICKS_PER_SECOND = TICKS_PER_QUARTER * 2.0

    private data class TimedEvent(val tick: Int, val priority: Int, val bytes: ByteArray)

    fun write(notes: List<MidiNote>): ByteArray {
        val melodicPrograms = notes.filterNot { it.isDrum }
            .map { it.program.coerceIn(0, 127) }
            .distinct()
            .sorted()
        val channels = melodicPrograms.mapIndexed { index, program ->
            program to melodicChannel(index)
        }.toMap()

        val tracks = ArrayList<ByteArray>()
        tracks += metaTrack()
        for (program in melodicPrograms) {
            tracks += noteTrack(
                name = "Program ${program + 1}",
                program = program,
                channel = channels.getValue(program),
                notes = notes.filter { !it.isDrum && it.program.coerceIn(0, 127) == program },
            )
        }
        val drums = notes.filter { it.isDrum }
        if (drums.isNotEmpty()) tracks += noteTrack("Drums", null, 9, drums)

        val output = ByteArrayOutputStream()
        output.writeAscii("MThd")
        output.writeInt32(6)
        output.writeInt16(1)
        output.writeInt16(tracks.size)
        output.writeInt16(TICKS_PER_QUARTER)
        tracks.forEach { track ->
            output.writeAscii("MTrk")
            output.writeInt32(track.size)
            output.write(track)
        }
        return output.toByteArray()
    }

    private fun metaTrack(): ByteArray {
        val out = ByteArrayOutputStream()
        writeVariableLength(out, 0)
        out.write(byteArrayOf(0xff.toByte(), 0x51, 0x03, 0x07, 0xa1.toByte(), 0x20))
        writeVariableLength(out, 0)
        out.write(byteArrayOf(0xff.toByte(), 0x58, 0x04, 0x04, 0x02, 0x18, 0x08))
        writeEndOfTrack(out)
        return out.toByteArray()
    }

    private fun noteTrack(
        name: String,
        program: Int?,
        channel: Int,
        notes: List<MidiNote>,
    ): ByteArray {
        val events = ArrayList<TimedEvent>()
        if (program != null) events += TimedEvent(0, 0, byteArrayOf((0xc0 or channel).toByte(), program.toByte()))
        for (note in notes) {
            val pitch = note.pitch.coerceIn(0, 127)
            val start = secondsToTicks(note.onsetSeconds)
            val end = secondsToTicks(note.offsetSeconds).coerceAtLeast(start + 1)
            events += TimedEvent(end, 0, byteArrayOf((0x80 or channel).toByte(), pitch.toByte(), 0))
            events += TimedEvent(start, 1, byteArrayOf((0x90 or channel).toByte(), pitch.toByte(), 96.toByte()))
        }
        events.sortWith(compareBy<TimedEvent> { it.tick }.thenBy { it.priority })

        val out = ByteArrayOutputStream()
        val nameBytes = name.toByteArray(Charsets.UTF_8)
        writeVariableLength(out, 0)
        out.write(byteArrayOf(0xff.toByte(), 0x03))
        writeVariableLength(out, nameBytes.size)
        out.write(nameBytes)

        var previousTick = 0
        for (event in events) {
            writeVariableLength(out, event.tick - previousTick)
            out.write(event.bytes)
            previousTick = event.tick
        }
        writeEndOfTrack(out)
        return out.toByteArray()
    }

    private fun secondsToTicks(seconds: Double): Int =
        (seconds.coerceAtLeast(0.0) * TICKS_PER_SECOND).roundToInt()

    private fun melodicChannel(index: Int): Int {
        val channel = index % 15
        return if (channel >= 9) channel + 1 else channel
    }

    private fun writeEndOfTrack(out: ByteArrayOutputStream) {
        writeVariableLength(out, 0)
        out.write(byteArrayOf(0xff.toByte(), 0x2f, 0x00))
    }

    private fun writeVariableLength(out: ByteArrayOutputStream, value: Int) {
        var current = value.coerceAtLeast(0)
        var buffer = current and 0x7f
        while (current ushr 7 > 0) {
            current = current ushr 7
            buffer = (buffer shl 8) or ((current and 0x7f) or 0x80)
        }
        while (true) {
            out.write(buffer and 0xff)
            if (buffer and 0x80 != 0) buffer = buffer ushr 8 else break
        }
    }

    private fun ByteArrayOutputStream.writeAscii(value: String) = write(value.toByteArray(Charsets.US_ASCII))
    private fun ByteArrayOutputStream.writeInt16(value: Int) {
        write((value ushr 8) and 0xff)
        write(value and 0xff)
    }
    private fun ByteArrayOutputStream.writeInt32(value: Int) {
        write((value ushr 24) and 0xff)
        write((value ushr 16) and 0xff)
        write((value ushr 8) and 0xff)
        write(value and 0xff)
    }
}
