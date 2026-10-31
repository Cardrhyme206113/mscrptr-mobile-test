package com.muscript.pocket;

import android.os.Debug;
import android.os.SystemClock;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/** Java ONNX Runtime implementation of the merged MuScriptor INT4 model. */
final class OnnxTranscriber {
    interface Events {
        void update(String message, int chunk, int chunks, int tokens, double seconds);
        void notes(List<MidiTracker.Note> notes);
        void info(String message);
    }
    static final int LAYERS=14, HEADS=12, HDIM=64, VOCAB=1393, BOS=1393, EOS=1;
    static final class Result {
        byte[] midi; int notes, tokens; double elapsed, tps; String backend;
    }
    static final class Settings {
        int maxSeconds=30,maxTokens=1000,threads=2;
        boolean prelude=true,nnapi=false;
    }
    private final File model;
    private final Events events;
    private final AtomicBoolean stop;
    private OrtSession residentSession;
    private OrtSession.SessionOptions residentOptions;
    private int residentThreads=-1;
    private boolean residentNnapi=false;
    private String residentBackend="CPU";
    OnnxTranscriber(File model,Events events,AtomicBoolean stop){this.model=model;this.events=events;this.stop=stop;}
    private static LongBuffer longs(int[] ids){
        LongBuffer buf=ByteBuffer.allocateDirect(ids.length*8).order(ByteOrder.nativeOrder()).asLongBuffer();
        for(int t:ids)buf.put((long)t);buf.rewind();return buf;
    }
    private static FloatBuffer floats(float[] data){
        FloatBuffer buf=ByteBuffer.allocateDirect(data.length*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        buf.put(data);buf.rewind();return buf;
    }
    private static FloatBuffer empty(){return ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asFloatBuffer();}
    private boolean shouldStop(){return stop.get()||Thread.currentThread().isInterrupted();}
    private int argmax(OnnxTensor tensor){
        FloatBuffer b=tensor.getFloatBuffer();
        int best=0;float top=-Float.MAX_VALUE;
        // first-step can return logits for multiple forced tokens; use last 1393 values.
        int start=b.limit()-VOCAB;
        if(start<0)throw new IllegalStateException("Logits smaller than vocabulary");
        for(int i=0;i<VOCAB;i++){
            float x=b.get(start+i);
            if(!Float.isFinite(x))throw new IllegalStateException("Nonfinite model logit");
            if(x>top){top=x;best=i;}
        }
        return best;
    }
    private void update(int ci,int nc,int count,double now,String message){
        events.update(message,ci,nc,count,now);
    }
    /** Keep the graph and native packed weights resident between runs. */
    private synchronized OrtSession ensureSession(OrtEnvironment env,Settings s) throws Exception{
        if(residentSession!=null&&residentThreads==s.threads&&residentNnapi==s.nnapi)return residentSession;
        close();
        OrtSession.SessionOptions opts=new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(s.threads);
        opts.setInterOpNumThreads(1);
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        try{
            if(s.nnapi)opts.addNnapi();
            residentSession=env.createSession(model.getAbsolutePath(),opts);
            residentOptions=opts;residentThreads=s.threads;residentNnapi=s.nnapi;
            residentBackend=s.nnapi?"NNAPI requested (+ CPU fallback)":"CPU";
            return residentSession;
        }catch(Exception e){
            opts.close();
            if(!s.nnapi)throw e;
            events.info("NNAPI unavailable, attempting CPU: "+e.getMessage());
            Settings cpu=new Settings();cpu.threads=s.threads;cpu.nnapi=false;
            return ensureSession(env,cpu);
        }
    }
    synchronized void close(){
        if(residentSession!=null){try{residentSession.close();}catch(Exception ignored){}residentSession=null;}
        if(residentOptions!=null){try{residentOptions.close();}catch(Exception ignored){}residentOptions=null;}
        residentThreads=-1;
    }
    Result run(AudioFrontend.AudioData audio, Settings s) throws Exception {
        long begin=SystemClock.elapsedRealtimeNanos();
        MidiTracker tracker=new MidiTracker();
        OrtEnvironment env=OrtEnvironment.getEnvironment();
        events.info(String.format(Locale.US,"Model %.1f MiB, %d thread(s)",model.length()/1048576.,s.threads));
        OrtSession runner=ensureSession(env,s);
        String backend=residentBackend;
        {
            int chunks=(int)Math.ceil(audio.audio.length/(double)AudioFrontend.CHUNK);
            int produced=0;
            events.info("Execution: "+backend+"; "+chunks+" chunk(s). This is NOT Vulkan.");
            for(int j=0;j<chunks;j++){
                if(shouldStop())throw new InterruptedException("Stopped");
                double seek=j*5.0;
                tracker.boundary(seek,j+1<chunks?(j+1)*5.0:Double.POSITIVE_INFINITY);
                int[] forced=j>0&&s.prelude?tracker.forcedPrelude():new int[0];
                if(forced.length>=s.maxTokens)throw new IllegalArgumentException("Forced prelude longer than token limit");
                update(j,chunks,produced,(SystemClock.elapsedRealtimeNanos()-begin)/1e9,"Preparing audio "+(j+1)+"/"+chunks);
                float[] mel=AudioFrontend.logMel(audio.audio,j*AudioFrontend.CHUNK);
                float[] mask=new float[AudioFrontend.FRAMES];
                int remaining=audio.audio.length-j*AudioFrontend.CHUNK;
                int validFrames=Math.min(500,(int)Math.ceil(Math.min(AudioFrontend.CHUNK,remaining)/(double)AudioFrontend.HOP));
                for(int i=0;i<validFrames;i++)mask[i]=1f;
                try (OnnxTensor melTensor=OnnxTensor.createTensor(env,floats(mel),new long[]{1,501,512});
                     OnnxTensor maskTensor=OnnxTensor.createTensor(env,floats(mask),new long[]{1,501});
                     OnnxTensor instrument=OnnxTensor.createTensor(env,longs(new int[]{-1}),new long[]{1,1});
                     OnnxTensor conditioned=OnnxTensor.createTensor(env,Boolean.TRUE);
                     OnnxTensor unconditioned=OnnxTensor.createTensor(env,Boolean.FALSE);
                     OnnxTensor emptyKeys=OnnxTensor.createTensor(env,empty(),new long[]{14,1,12,0,64});
                     OnnxTensor emptyVals=OnnxTensor.createTensor(env,empty(),new long[]{14,1,12,0,64})) {
                    OrtSession.Result previous=null;
                    long input[]=new long[1+forced.length];input[0]=BOS;
                    for(int f=0;f<forced.length;f++){input[f+1]=forced[f];tracker.token(forced[f]);}
                    boolean eos=false;
                    try{
                        for(int i=0;i<s.maxTokens-forced.length;i++){
                            if(shouldStop())throw new InterruptedException("Stopped");
                            int[] tokenIds=new int[input.length];for(int p=0;p<input.length;p++)tokenIds[p]=(int)input[p];
                            try (OnnxTensor ids=OnnxTensor.createTensor(env,longs(tokenIds),new long[]{1,input.length})) {
                                Map<String,OnnxTensor> feeds=new HashMap<>();
                                feeds.put("token_ids",ids);
                                feeds.put("past_keys",previous==null?emptyKeys:(OnnxTensor)previous.get("present_keys").orElseThrow());
                                feeds.put("past_values",previous==null?emptyVals:(OnnxTensor)previous.get("present_values").orElseThrow());
                                feeds.put("logmel",melTensor);feeds.put("mel_mask",maskTensor);
                                feeds.put("instrument_ids",instrument);
                                feeds.put("use_conditioning",previous==null?conditioned:unconditioned);
                                OrtSession.Result next=runner.run(feeds);
                                if(previous!=null)previous.close();previous=next;
                                int token=argmax((OnnxTensor)next.get("logits").orElseThrow());
                                produced++;
                                if(token==EOS){eos=true;break;}
                                tracker.token(token);
                                input=new long[]{token};
                                if(i%12==11){
                                    List<MidiTracker.Note> notes=tracker.drainNew();
                                    if(!notes.isEmpty())events.notes(notes);
                                    update(j,chunks,produced,(SystemClock.elapsedRealtimeNanos()-begin)/1e9,
                                           "Chunk "+(j+1)+"/"+chunks+" · token "+(i+1));
                                }
                            }
                        }
                    } finally { if(previous!=null)previous.close(); }
                    List<MidiTracker.Note> notes=tracker.drainNew();if(!notes.isEmpty())events.notes(notes);
                    events.info("Chunk "+(j+1)+"/"+chunks+(eos?" completed":" reached token limit"));
                }
            }
            tracker.finish(audio.duration);events.notes(tracker.drainNew());
            Result result=new Result();
            result.midi=tracker.midi();result.notes=tracker.notes.size();result.tokens=produced;
            result.elapsed=(SystemClock.elapsedRealtimeNanos()-begin)/1e9;
            result.tps=produced/Math.max(.01,result.elapsed);
            result.backend=backend;return result;
        }
    }
}
