package com.muscript.pocket;
/** Standalone (no Android dependencies) test: javac ...MidiTracker.java ...DecoderSelfTest.java */
public final class DecoderSelfTest {
    public static void main(String[] args) throws Exception {
        MidiTracker t=new MidiTracker();
        t.boundary(0,5);
        t.token(1134);  // tie prelude terminator
        t.token(1135);  // Program 0
        t.token(1133);  // velocity = on
        t.token(1064);  // pitch 60
        t.token(53);    // time shift 50 -> .50 seconds
        t.token(1132);  // velocity off
        t.token(1064);  // note off
        t.finish(5);
        System.out.println("notes="+t.notes.size()+ (t.notes.isEmpty()?"":(" pitch="+t.notes.get(0).pitch+" start="+t.notes.get(0).start+" end="+t.notes.get(0).end)));
        if(t.notes.size()!=1 ||t.notes.get(0).pitch!=60 ||Math.abs(t.notes.get(0).end-.5)>.001) throw new AssertionError("Token decoding failed");
        byte[] midi=t.midi();
        String s=new String(midi,0,4,java.nio.charset.StandardCharsets.US_ASCII);
        if(!"MThd".equals(s)||midi.length<40)throw new AssertionError("MIDI header invalid");
        System.out.println("PASS: token events and MIDI file ("+midi.length+" bytes)");
        java.nio.file.Files.write(java.nio.file.Path.of("/mnt/data/selftest.mid"),midi);
    }
}
