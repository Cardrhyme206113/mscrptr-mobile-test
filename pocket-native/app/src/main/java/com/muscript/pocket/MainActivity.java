package com.muscript.pocket;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Debug;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.view.Window;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.WebSettings;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Original MuScript Pocket HTML UI + real native ONNX Runtime inference. */
public final class MainActivity extends Activity implements OnnxTranscriber.Events {
    private static final int OPEN_AUDIO=70,SAVE_MIDI=71,OPEN_MODEL=72;
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancel=new AtomicBoolean(false);
    private WebView web;
    private Uri audioUri;
    private String audioName="audio";
    private byte[] lastMidi;
    private boolean running=false;
    private OnnxTranscriber engine;
    @Override public void onCreate(Bundle saved){
        super.onCreate(saved);
        getWindow().setStatusBarColor(0xff0d0e10);getWindow().setNavigationBarColor(0xff0d0e10);
        web=new WebView(this);web.setBackgroundColor(0xff0d0e10);
        WebSettings settings=web.getSettings();settings.setJavaScriptEnabled(true);
        settings.setAllowFileAccess(false);settings.setAllowContentAccess(false);
        settings.setDomStorageEnabled(false);settings.setMediaPlaybackRequiresUserGesture(true);
        web.setWebViewClient(new WebViewClient(){
            @Override public boolean shouldOverrideUrlLoading(WebView v,android.webkit.WebResourceRequest r){return true;}
        });
        web.addJavascriptInterface(new Bridge(),"Native");
        setContentView(web);
        web.loadUrl("file:///android_asset/index.html");
    }
    private void js(String code){runOnUiThread(()->{if(web!=null)web.evaluateJavascript(code,null);});}
    private static String json(String s){return JSONObject.quote(s);}
    private File modelFile() throws Exception {
        File target=new File(getFilesDir(),"muscriptor-small-combined-int4.onnx");
        if(target.exists()&&target.length()>70_000_000L)return target;
        throw new IllegalStateException("Model not installed. Tap Generate and select your combined INT4 ONNX file first.");
    }
    private void chooseModel(){
        Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent,OPEN_MODEL);
    }
    private String filename(Uri uri){
        try(Cursor cursor=getContentResolver().query(uri,new String[]{OpenableColumns.DISPLAY_NAME},null,null,null)){
            if(cursor!=null&&cursor.moveToFirst())return cursor.getString(0);
        }catch(Exception ignored){}
        return "audio";
    }
    private void chooseAudio(){
        Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);intent.setType("audio/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent,OPEN_AUDIO);
    }
    private void exportMidi(){
        if(lastMidi==null)return;
        Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);intent.setType("audio/midi");
        String title=audioName.replaceAll("\\.[^.]*$","")+"-muscriptor-int4.mid";
        intent.putExtra(Intent.EXTRA_TITLE,title);
        startActivityForResult(intent,SAVE_MIDI);
    }
    @Override protected void onActivityResult(int request,int result,Intent data){
        super.onActivityResult(request,result,data);
        if(result!=RESULT_OK||data==null||data.getData()==null)return;
        if(request==OPEN_AUDIO){
            audioUri=data.getData();audioName=filename(audioUri);
            try{getContentResolver().takePersistableUriPermission(audioUri,Intent.FLAG_GRANT_READ_URI_PERMISSION);}catch(Exception ignored){}
            lastMidi=null;
            js("window.nativeAudioPicked("+json(audioName)+");");
        }else if(request==OPEN_MODEL){
            final Uri modelUri=data.getData();
            info("Importing ONNX model…");
            worker.submit(()->{
                File target=new File(getFilesDir(),"muscriptor-small-combined-int4.onnx");
                File tmp=new File(getFilesDir(),"model-incoming.tmp");
                try(InputStream in=getContentResolver().openInputStream(modelUri);
                    FileOutputStream out=new FileOutputStream(tmp)){
                    if(in==null)throw new IllegalStateException("Cannot read model file");
                    byte[] buffer=new byte[1024*1024]; int count;
                    while((count=in.read(buffer))!=-1)out.write(buffer,0,count);
                    out.flush();
                    if(tmp.length()<70_000_000L)throw new IllegalArgumentException("File too small; select combined INT4 model (~69 MiB)");
                    if(target.exists()&&!target.delete())throw new IllegalStateException("Cannot replace old model");
                    if(!tmp.renameTo(target))throw new IllegalStateException("Model installation failed");
                    if(engine!=null){engine.close();engine=null;}
                    info("ONNX installed ("+String.format(Locale.US,"%.1f",target.length()/1048576.0)+" MiB). Tap Generate again.");
                }catch(Exception ex){tmp.delete();info("Model import failed: "+ex.getMessage());}
            });
        }else if(request==SAVE_MIDI&&lastMidi!=null){
            try(OutputStream out=getContentResolver().openOutputStream(data.getData())){
                if(out==null)throw new IllegalStateException("Cannot open destination");
                out.write(lastMidi);out.flush();
                js("window.nativeStatus('MIDI saved successfully');");
            }catch(Exception e){info("MIDI save failed: "+e.getMessage());}
        }
    }
    private void start(int maxSeconds,int tokens,int threads,boolean prelude,boolean nnapi){
        if(audioUri==null){info("Select an audio file first");return;}
        if(running){info("Inference is already running");return;}
        File ready=new File(getFilesDir(),"muscriptor-small-combined-int4.onnx");
        if(!ready.exists()||ready.length()<70_000_000L){
            info("Select your muscriptor-small-combined-int4.onnx file (first run)");
            chooseModel();return;
        }
        running=true;lastMidi=null;cancel.set(false);
        final Uri selection=audioUri;
        final OnnxTranscriber.Settings cfg=new OnnxTranscriber.Settings();
        cfg.maxSeconds=Math.min(180,Math.max(0,maxSeconds));
        cfg.maxTokens=Math.min(3000,Math.max(32,tokens));
        cfg.threads=Math.min(8,Math.max(1,threads));cfg.prelude=prelude;cfg.nnapi=nnapi;
        js("window.nativeStarted();");
        worker.submit(()->{
            try{
                info("Decoding audio on device…");
                AudioFrontend.AudioData audio=AudioFrontend.decode(getContentResolver(),selection,cfg.maxSeconds);
                info(String.format(Locale.US,"Audio decoded: %.1f s / 16 kHz",audio.duration));
                if(cancel.get())throw new InterruptedException("Stopped");
                if(engine==null)engine=new OnnxTranscriber(modelFile(),this,cancel);
                OnnxTranscriber.Result output=engine.run(audio,cfg);
                lastMidi=output.midi;
                double pss=0;Debug.MemoryInfo mem=new Debug.MemoryInfo();Debug.getMemoryInfo(mem);pss=mem.getTotalPss()/1024.;
                js("window.nativeCompleted("+output.notes+","+output.tokens+","+
                        String.format(Locale.US,"%.2f",output.elapsed)+","+
                        String.format(Locale.US,"%.2f",output.tps)+","+
                        String.format(Locale.US,"%.1f",pss)+");");
            }catch(InterruptedException e){js("window.nativeFailed('Stopped by user');");}
            catch(Throwable t){
                android.util.Log.e("MuScriptPocket","Inference failure",t);
                js("window.nativeFailed("+json(t.getClass().getSimpleName()+": "+t.getMessage())+");");
            }finally{running=false;}
        });
    }
    @Override public void update(String msg,int chunk,int count,int tokens,double elapsed){
        js("window.nativeProgress("+json(msg)+","+chunk+","+count+","+tokens+","+
                String.format(Locale.US,"%.2f",elapsed)+");");
    }
    @Override public void info(String message){js("window.nativeStatus("+json(message)+");");}
    @Override public void notes(List<MidiTracker.Note> notes){
        if(notes.isEmpty())return;
        JSONArray a=new JSONArray();
        try{
            for(MidiTracker.Note n:notes){
                JSONObject obj=new JSONObject();obj.put("pitch",n.pitch);obj.put("program",n.program);
                obj.put("start",n.start);obj.put("end",n.end);a.put(obj);
            }
        }catch(Exception ignored){}
        js("window.nativeNotes("+a.toString()+");");
    }
    private final class Bridge {
        @JavascriptInterface public void pickAudio(){runOnUiThread(()->chooseAudio());}
        @JavascriptInterface public void pickModel(){runOnUiThread(()->chooseModel());}
        @JavascriptInterface public void clearAudio(){audioUri=null;lastMidi=null;}
        @JavascriptInterface public void saveMidi(){runOnUiThread(()->exportMidi());}
        @JavascriptInterface public void stop(){cancel.set(true);}
        @JavascriptInterface public void run(int secs,int tokens,int threads,boolean prelude,boolean nnapi){
            runOnUiThread(()->start(secs,tokens,threads,prelude,nnapi));
        }
    }
    @Override protected void onDestroy(){
        cancel.set(true);worker.shutdownNow();
        if(!running&&engine!=null)engine.close();
        if(web!=null){web.removeJavascriptInterface("Native");web.destroy();web=null;}
        super.onDestroy();
    }
}
