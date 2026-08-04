package dev.cardrhyme.muscriptormobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MidiWriterTest {
    @Test
    fun writesValidHeaderAndTrackCount() {
        val midi = MidiWriter.write(
            listOf(
                MidiNote(0, 60, 0.0, 0.5, false),
                MidiNote(128, 36, 0.25, 0.26, true),
            ),
        )
        assertTrue(midi.size > 20)
        assertEquals("MThd", midi.copyOfRange(0, 4).toString(Charsets.US_ASCII))
        assertEquals(3, ((midi[10].toInt() and 0xff) shl 8) or (midi[11].toInt() and 0xff))
    }
}
