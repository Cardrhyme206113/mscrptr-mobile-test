package com.muscript.pocket;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** MT3 event token reconstruction. Decodes notes as the ONNX token stream arrives. */
final class MidiTracker {
    static final class Note {
        int program,pitch; double start,end; boolean drum;
        Note(int program,int pitch,double start,boolean drum){this.program=program;this.pitch=pitch;this.start=start;this.drum=drum;}
    }
    final List<Note> notes=new ArrayList<>();
    private final Map<String,Note> open=new LinkedHashMap<>();
    private final java.util.HashSet<String> ties=new java.util.HashSet<>();
    private boolean started=false,inPrelude=true,skip=false;
    private int program=-1,velocity=-1, tick=0,startTick=0;
    private double seek=0,next=Double.POSITIVE_INFINITY;
    int noteIndex=0;
    private String key(int pr,int p){return pr+":"+p;}
    private void close(String k,double time){
        Note n=open.remove(k);
        if(n!=null){n.end=Math.max(n.start+.01,time); notes.add(n);}
    }
    void boundary(double start,double end){
        if(started&&inPrelude)for(String k:new ArrayList<>(open.keySet()))close(k,seek);
        started=true; seek=start;next=end;tick=startTick=(int)Math.round(start*100);
        program=-1;velocity=-1;inPrelude=true;skip=false;ties.clear();
    }
    void token(int t){
        if(t<3)return;
        int type=-1,v=0;
        if(t<=1003){type=0;v=t-3;}
        else if(t<=1131){type=1;v=t-1004;}
        else if(t<=1133){type=2;v=t-1132;}
        else if(t==1134){type=3;}
        else if(t<=1264){type=4;v=t-1135;}
        else if(t<=1392){type=5;v=t-1265;}
        if(inPrelude){
            if(type==3){
                inPrelude=false;velocity=-1;
                for(String k:new ArrayList<>(open.keySet()))if(!ties.contains(k))close(k,seek);
            }else if(type==0){
                inPrelude=false;skip=true;
                for(String k:new ArrayList<>(open.keySet()))close(k,seek);
            }else if(type==4) program=v;
            else if(type==1&&program>=0)ties.add(key(program,v));
            return;
        }
        if(skip)return;
        switch(type){
            case 0: if(v>0)tick=startTick+v;break;
            case 4: program=v;break;
            case 2: velocity=v;break;
            case 5:{
                double time=tick/100.0;
                if(time<next){Note n=new Note(128,v,time,true);n.end=time+.01;notes.add(n);}
                break;
            }
            case 1:{
                if(program<0||velocity<0)break;
                double time=tick/100.0;if(time>=next)break;
                String k=key(program,v);
                if(open.containsKey(k))close(k,time);
                if(velocity>0)open.put(k,new Note(program,v,time,false));
                break;
            }
        }
    }
    int[] forcedPrelude(){
        ArrayList<Note> held=new ArrayList<>(open.values());
        held.sort(Comparator.comparingInt((Note n)->n.program).thenComparingInt(n->n.pitch));
        ArrayList<Integer> ids=new ArrayList<>();int prev=-1;
        for(Note n:held){
            if(n.program!=prev){ids.add(1135+n.program);prev=n.program;}
            ids.add(1004+n.pitch);
        }
        ids.add(1134);
        int[] a=new int[ids.size()];for(int i=0;i<a.length;i++)a[i]=ids.get(i);return a;
    }
    void finish(double duration){for(String k:new ArrayList<>(open.keySet()))close(k,duration);}
    List<Note> drainNew(){
        ArrayList<Note> delta=new ArrayList<>(notes.subList(noteIndex,notes.size()));
        noteIndex=notes.size();return delta;
    }
    static final class Event {
        long time; int priority,key; byte[] bytes;
        Event(double sec,int priority,int key,int... vals){
            this.time=Math.max(0,Math.round(sec*960));this.priority=priority;this.key=key;
            this.bytes=new byte[vals.length];for(int i=0;i<vals.length;i++)this.bytes[i]=(byte)vals[i];
        }
    }
    private static void text(ByteArrayOutputStream b,String s){for(int i=0;i<s.length();i++)b.write((byte)s.charAt(i));}
    private static void int32(ByteArrayOutputStream b,long n){b.write((int)(n>>24)&255);b.write((int)(n>>16)&255);b.write((int)(n>>8)&255);b.write((int)n&255);}
    private static void int16(ByteArrayOutputStream b,int n){b.write((n>>8)&255);b.write(n&255);}
    private static void varLen(ByteArrayOutputStream b,long n){
        if(n<0)n=0; if(n>0x0FFFFFFFL)n=0x0FFFFFFFL;
        int[] bytes=new int[5];int count=0;bytes[count++]=(int)(n&127);n>>=7;
        while(n!=0){bytes[count++]=(int)(0x80|(n&127));n>>=7;}
        for(int i=count-1;i>=0;i--)b.write(bytes[i]);
    }
    private static byte[] track(List<Event> events){
        events.sort(Comparator.comparingLong((Event e)->e.time).thenComparingInt(e->e.priority).thenComparingInt(e->e.key));
        ByteArrayOutputStream raw=new ByteArrayOutputStream();long last=0;
        for(Event e:events){long when=Math.max(e.time,last);varLen(raw,when-last);raw.write(e.bytes,0,e.bytes.length);last=when;}
        raw.write(0);raw.write(255);raw.write(47);raw.write(0);
        ByteArrayOutputStream dst=new ByteArrayOutputStream();text(dst,"MTrk");int32(dst,raw.size());byte[] data=raw.toByteArray();dst.write(data,0,data.length);return dst.toByteArray();
    }
    byte[] midi(){
        Map<Integer,List<Note>> groups=new LinkedHashMap<>();
        for(Note n:notes){
            if(n.pitch<0||n.pitch>127||n.program<0||n.program>128||!Double.isFinite(n.end)||!Double.isFinite(n.start))continue;
            groups.computeIfAbsent(n.program,p->new ArrayList<>()).add(n);
        }
        ByteArrayOutputStream output=new ByteArrayOutputStream();
        text(output,"MThd");int32(output,6);int16(output,1);int16(output,groups.size()+1);int16(output,480);
        List<Event> tempo=new ArrayList<>();tempo.add(new Event(0,0,0,0xFF,0x51,0x03,0x07,0xA1,0x20));
        byte[] part=track(tempo);output.write(part,0,part.length);
        final int[] channels={0,1,2,3,4,5,6,7,8,10,11,12,13,14,15};int channelIndex=0;
        for(Map.Entry<Integer,List<Note>> entry:groups.entrySet()){
            int program=entry.getKey(),ch=program==128?9:channels[(channelIndex++)%channels.length];
            List<Event> events=new ArrayList<>();
            if(program!=128)events.add(new Event(0,0,0,0xC0|ch,program));
            for(Note n:entry.getValue()){
                events.add(new Event(n.start,2,n.pitch,0x90|ch,n.pitch,program==128?96:85));
                events.add(new Event(Math.max(n.start+.01,n.end),1,n.pitch,0x80|ch,n.pitch,0));
            }
            part=track(events);output.write(part,0,part.length);
        }
        return output.toByteArray();
    }
}
