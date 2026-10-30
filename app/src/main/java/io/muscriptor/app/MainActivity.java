package io.muscriptor.app;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.content.Context;
import android.net.Uri;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.AudioFormat;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.WebChromeClient;
import android.view.WindowManager;
import android.graphics.Color;
import android.os.Build;
import android.provider.OpenableColumns;
import android.database.Cursor;
import android.content.res.AssetManager;

import org.json.JSONObject;
import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import android.util.Base64;

public final class MainActivity extends Activity {
    private static final int PICK_AUDIO = 71, PICK_MODEL = 72, SAVE_MIDI = 73;
    private WebView view;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Uri audioUri;
    private File modelFile;
    private byte[] pendingMidi;
    private boolean busy = false;
    private String audioName = "song.mp3";
    private static final boolean JNI_LOADED;
    static {
        boolean loaded;
        try { System.loadLibrary("muscriptor_jni"); loaded = true; }
        catch (UnsatisfiedLinkError e) { loaded = false; }
        JNI_LOADED = loaded;
    }

    private native String nativeTranscribe(String modelPath, float[] samples, boolean prelude, boolean pianoOnly, boolean repair);
    private native void nativeCancel();

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(Color.rgb(16,19,24));
        getWindow().setNavigationBarColor(Color.rgb(16,19,24));
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
        view = new WebView(this);
        view.setBackgroundColor(Color.rgb(16,19,24));
        view.getSettings().setJavaScriptEnabled(true);
        view.getSettings().setDomStorageEnabled(true);
        view.getSettings().setAllowFileAccess(true);
        view.getSettings().setAllowFileAccessFromFileURLs(false);
        view.getSettings().setAllowUniversalAccessFromFileURLs(false);
        view.getSettings().setBuiltInZoomControls(false);
        view.getSettings().setDisplayZoomControls(false);
        modelFile = new File(getFilesDir(), "muscriptor-small-f16.gguf");
        view.setWebViewClient(new WebViewClient() {
            @Override public void onPageFinished(WebView web, String url) {
                if (modelFile.isFile() && modelFile.length() > 10000000)
                    js("window.nativeModelSelected('muscriptor-small-f16.gguf',true)");
            }
        });
        view.setWebChromeClient(new WebChromeClient());
        view.addJavascriptInterface(new BrowserBridge(), "MuScriptorAndroid");
        setContentView(view);
        view.loadUrl("file:///android_asset/index.html");
    }

    private void js(String js) { runOnUiThread(() -> { if (view != null) view.evaluateJavascript(js, null); }); }
    private static String quoted(String str) { return JSONObject.quote(str); }
    private String filename(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[] { OpenableColumns.DISPLAY_NAME }, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (Exception ignored) {}
        String seg = uri.getLastPathSegment(); return seg == null ? "song.mp3" : seg;
    }
    public final class BrowserBridge {
        @JavascriptInterface public void pickAudio() {
            runOnUiThread(() -> {
                Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("*/*");
                i.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/midi", "audio/x-midi", "application/octet-stream"});
                startActivityForResult(i, PICK_AUDIO);
            });
        }
        @JavascriptInterface public void pickModel() {
            runOnUiThread(() -> {
                Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("*/*");
                startActivityForResult(i, PICK_MODEL);
            });
        }
        @JavascriptInterface public void transcribe(boolean prelude, boolean pianoOnly, boolean repair) {
            runOnUiThread(() -> {
                if (busy) return;
                if (audioUri == null) { js("window.nativeTranscriptionError('Choose an MP3 first')"); return; }
                if (!modelFile.isFile() || modelFile.length() < 10000000) {
                    js("window.nativeTranscriptionError('Choose a valid GGUF model in Settings')"); return;
                }
                if (!JNI_LOADED) { js("window.nativeTranscriptionError('Native inference library not installed in this build')"); return; }
                busy = true;
                js("window.nativeTranscriptionStarted()");
                worker.execute(() -> {
                    try {
                        float[] samples = decodeAudio(audioUri);
                        if (samples.length == 0) throw new IOException("No audio decoded");
                        String result = nativeTranscribe(modelFile.getAbsolutePath(), samples, prelude, pianoOnly, repair);
                        js("window.nativeTranscriptionComplete(" + quoted(result) + ")");
                    } catch (Throwable e) {
                        js("window.nativeTranscriptionError(" + quoted(String.valueOf(e.getMessage())) + ")");
                    } finally { busy = false; }
                });
            });
        }
        @JavascriptInterface public void saveMidi(String base64, String filename) {
            try {
                pendingMidi = Base64.decode(base64, Base64.DEFAULT);
                runOnUiThread(() -> {
                    Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                    i.addCategory(Intent.CATEGORY_OPENABLE);
                    i.setType("audio/midi");
                    i.putExtra(Intent.EXTRA_TITLE, filename);
                    startActivityForResult(i, SAVE_MIDI);
                });
            } catch (Exception e) { js("window.nativeTranscriptionError('Could not export MIDI')"); }
        }
    }

    @Override protected void onActivityResult(int code, int result, Intent intent) {
        super.onActivityResult(code, result, intent);
        if (result != RESULT_OK || intent == null || intent.getData() == null) return;
        Uri uri = intent.getData();
        if (code == SAVE_MIDI) {
            try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                if (out != null && pendingMidi != null) out.write(pendingMidi);
                js("window.nativeExportSaved && window.nativeExportSaved()");
            } catch (Exception e) { js("window.nativeTranscriptionError('Could not write MIDI')"); }
            pendingMidi = null; return;
        }
        String name = filename(uri);
        if (code == PICK_AUDIO) {
            String lower = name.toLowerCase(Locale.ROOT);
            if (!(lower.endsWith(".mp3") || lower.endsWith(".wav") || lower.endsWith(".mid") || lower.endsWith(".midi"))) {
                js("window.nativeTranscriptionError('Choose MP3, WAV, or MIDI')"); return;
            }
            audioName = name;
            if (lower.endsWith(".mid") || lower.endsWith(".midi")) {
                worker.execute(() -> {
                    try (InputStream in = getContentResolver().openInputStream(uri); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
                        if (in == null) throw new IOException("Could not read MIDI");
                        byte[] buffer = new byte[8192]; int n;
                        while ((n = in.read(buffer)) != -1) out.write(buffer,0,n);
                        String b64 = Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP);
                        js("window.nativeAudioSelected("+quoted(name)+",'midi',"+quoted(b64)+")");
                    } catch (Exception e) { js("window.nativeTranscriptionError('Unable to import MIDI')"); }
                });
            } else {
                audioUri = uri;
                js("window.nativeAudioSelected("+quoted(name)+",'audio','')");
            }
        } else if (code == PICK_MODEL) {
            if (!name.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
                js("window.nativeModelSelected("+quoted(name)+",false)"); return;
            }
            js("window.nativeTranscriptionError('Importing model…')");
            worker.execute(() -> {
                File staging = new File(getFilesDir(), "model.tmp");
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out = new BufferedOutputStream(new FileOutputStream(staging))) {
                    if (in == null) throw new IOException("Cannot open model");
                    byte[] block = new byte[1024*1024]; int n;
                    while ((n=in.read(block))!=-1) out.write(block,0,n);
                    out.flush();
                    if (staging.length() < 10000000) throw new IOException("Model file is too small");
                    if (modelFile.exists() && !modelFile.delete()) throw new IOException("Old model could not be deleted");
                    if (!staging.renameTo(modelFile)) throw new IOException("Model import failed");
                    js("window.nativeModelSelected("+quoted(name)+",true)");
                } catch (Exception e) {
                    staging.delete(); js("window.nativeModelSelected("+quoted(name)+",false)");
                }
            });
        }
    }

    /** Called on the transcription worker thread by the native library after each 5s chunk. */
    public void onNativeProgress(String noteJson, float progress, double finalizedThrough) {
        js("window.nativeTranscriptionProgress("+quoted(noteJson)+","+progress+","+finalizedThrough+")");
    }

    /** Android MediaCodec audio decode, downmix, then windowed-sinc resample to 16 kHz mono f32. */
    private float[] decodeAudio(Uri uri) throws Exception {
        MediaExtractor extractor = new MediaExtractor(); MediaCodec codec = null;
        try {
            extractor.setDataSource(this, uri, null);
            int track = -1; String mime = null; MediaFormat input = null;
            for (int i=0;i<extractor.getTrackCount();i++) {
                MediaFormat f=extractor.getTrackFormat(i);String m=f.getString(MediaFormat.KEY_MIME);
                if (m!=null && m.startsWith("audio/")) {track=i;mime=m;input=f;break;}
            }
            if (track<0) throw new IOException("No audio track");
            extractor.selectTrack(track);
            codec=MediaCodec.createDecoderByType(mime);
            codec.configure(input,null,null,0);codec.start();
            FloatBuilder pcm = new FloatBuilder();
            MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();
            boolean inputDone=false, outputDone=false;
            int sampleRate=input.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            int channels=input.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            int attempts=0;
            while (!outputDone) {
                if (!inputDone) {
                    int index=codec.dequeueInputBuffer(10000);
                    if (index>=0) {
                        ByteBuffer buffer=codec.getInputBuffer(index);if(buffer==null)throw new IOException("Missing input buffer");
                        buffer.clear();int size=extractor.readSampleData(buffer,0);
                        if (size<0){codec.queueInputBuffer(index,0,0,0,MediaCodec.BUFFER_FLAG_END_OF_STREAM);inputDone=true;}
                        else {codec.queueInputBuffer(index,0,size,extractor.getSampleTime(),0);extractor.advance();}
                    }
                }
                int index=codec.dequeueOutputBuffer(info,10000);
                if (index==MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat format=codec.getOutputFormat();
                    if (format.containsKey(MediaFormat.KEY_SAMPLE_RATE))sampleRate=format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT))channels=format.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                } else if (index>=0) {
                    ByteBuffer buf=codec.getOutputBuffer(index);
                    if (buf!=null && info.size>0) {
                        MediaFormat format=codec.getOutputFormat();
                        int enc=format.containsKey(MediaFormat.KEY_PCM_ENCODING) ? format.getInteger(MediaFormat.KEY_PCM_ENCODING) : AudioFormat.ENCODING_PCM_16BIT;
                        buf.order(ByteOrder.LITTLE_ENDIAN); buf.position(info.offset);buf.limit(info.offset+info.size);
                        int bytes=enc==AudioFormat.ENCODING_PCM_FLOAT?4:2;
                        int count=info.size/(bytes*channels);
                        for(int i=0;i<count;i++) {
                            double sum=0;
                            for(int c=0;c<channels;c++)sum+=(bytes==4?buf.getFloat():buf.getShort()/32768f);
                            pcm.add((float)(sum/channels));
                        }
                    }
                    codec.releaseOutputBuffer(index,false);
                    if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM)!=0) outputDone=true;
                } else if(++attempts>200000)throw new IOException("Audio decoder stalled");
            }
            return resample(pcm.array(), sampleRate, 16000);
        } finally {
            if(codec!=null){try{codec.stop();}catch(Exception ignored){}codec.release();}
            extractor.release();
        }
    }
    private static float[] resample(float[] src, int inRate, int outRate) {
        if(inRate==outRate)return src;
        int count=(int)Math.ceil(src.length*(double)outRate/inRate);
        float[] out=new float[count];
        double ratio=(double)inRate/outRate, cutoff=Math.min(1.0,(double)outRate/inRate)*.96;
        for(int i=0;i<count;i++) {
            double x=i*ratio, sum=0, weight=0;int center=(int)Math.floor(x);
            for(int j=-12;j<=12;j++) {
                int k=center+j;if(k<0||k>=src.length)continue;
                double d=x-k, z=Math.PI*d*cutoff;
                double sinc=Math.abs(z)<1e-8?1:Math.sin(z)/z;
                double window=.5+.5*Math.cos(Math.PI*d/13);
                double w=sinc*window*cutoff;sum+=src[k]*w;weight+=w;
            }
            out[i]=(float)(weight==0?0:sum/weight);
        }
        return out;
    }
    private static final class FloatBuilder {
        private float[] a = new float[65536];private int size=0;
        void add(float v){if(size>=a.length)a=Arrays.copyOf(a,a.length*2);a[size++]=v;}
        float[] array(){return Arrays.copyOf(a,size);}
    }
    @Override public void onDestroy() { if(busy && JNI_LOADED)nativeCancel();worker.shutdown();if(view!=null)view.destroy();super.onDestroy(); }
}
