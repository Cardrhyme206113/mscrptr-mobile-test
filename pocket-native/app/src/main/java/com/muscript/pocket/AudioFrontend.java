package com.muscript.pocket;

import android.content.ContentResolver;
import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/** Native audio decoding and faithful MuScriptor 16kHz / 512 HTK-mel magnitude frontend.
 * Audio resampling is linear (not torchaudio's bandlimited sinc), so some logits may differ.
 */
final class AudioFrontend {
    static final int RATE = 16000, CHUNK = 80000, FFT = 2048, HOP = 160, MEL = 512, FRAMES = 501;
    private static final int MAX_DURATION_SECONDS = 180;

    static final class AudioData {
        final float[] audio;
        final double duration;
        AudioData(float[] audio) { this.audio=audio; this.duration=audio.length/(double)RATE; }
    }
    private static final class FloatList {
        float[] a=new float[65536]; int n=0;
        void add(float v) { if(n==a.length) a=Arrays.copyOf(a,a.length*2); a[n++]=v; }
        float[] toArray(){ return Arrays.copyOf(a,n); }
    }
    static AudioData decode(ContentResolver resolver, Uri uri, int maxSeconds) throws Exception {
        MediaExtractor extractor=new MediaExtractor(); MediaCodec codec=null;
        try (ParcelFileDescriptor descriptor=resolver.openFileDescriptor(uri,"r")) {
            if(descriptor==null)throw new IOException("Cannot open selected audio");
            extractor.setDataSource(descriptor.getFileDescriptor());
            int track=-1;
            for(int i=0;i<extractor.getTrackCount();i++){
                MediaFormat f=extractor.getTrackFormat(i);
                if(f.getString(MediaFormat.KEY_MIME).startsWith("audio/")){ track=i; break; }
            }
            if(track<0) throw new IOException("No audio track found");
            extractor.selectTrack(track);
            MediaFormat sourceFormat=extractor.getTrackFormat(track);
            int channels=sourceFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            int sampleRate=sourceFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            String mime=sourceFormat.getString(MediaFormat.KEY_MIME);
            codec=MediaCodec.createDecoderByType(mime);
            codec.configure(sourceFormat,null,null,0); codec.start();
            int limitSeconds=maxSeconds>0?Math.min(maxSeconds,MAX_DURATION_SECONDS):MAX_DURATION_SECONDS;
            long limitUs=(long)limitSeconds*1_000_000L;
            FloatList pcm=new FloatList();
            boolean inputDone=false,outputDone=false;
            MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();
            int stalls=0,pcmEncoding=AudioFormat.ENCODING_PCM_16BIT;
            while(!outputDone){
                if(!inputDone){
                    int index=codec.dequeueInputBuffer(10000);
                    if(index>=0){
                        ByteBuffer buf=codec.getInputBuffer(index);
                        int len=extractor.readSampleData(buf,0);
                        long pts=extractor.getSampleTime();
                        if(len<0||pts>=limitUs){
                            codec.queueInputBuffer(index,0,0,Math.max(0,pts),MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inputDone=true;
                        }else{
                            codec.queueInputBuffer(index,0,len,pts,0); extractor.advance();
                        }
                    }
                }
                int index=codec.dequeueOutputBuffer(info,10000);
                if(index>=0){
                    stalls=0;
                    ByteBuffer buf=codec.getOutputBuffer(index);
                    if(buf!=null&&info.size>0){
                        buf.order(ByteOrder.LITTLE_ENDIAN);buf.position(info.offset);buf.limit(info.offset+info.size);
                        int frames=info.size/(channels*(pcmEncoding==AudioFormat.ENCODING_PCM_FLOAT?4:2));
                        int maximumFrames=(int)Math.ceil(limitSeconds*(double)sampleRate);
                        int append=Math.min(frames,Math.max(0,maximumFrames-pcm.n));
                        for(int frame=0;frame<append;frame++){
                            float sum=0;
                            for(int c=0;c<channels;c++) sum+=pcmEncoding==AudioFormat.ENCODING_PCM_FLOAT?buf.getFloat():buf.getShort()/32768f;
                            pcm.add(sum/channels);
                        }
                    }
                    outputDone=(info.flags&MediaCodec.BUFFER_FLAG_END_OF_STREAM)!=0;
                    codec.releaseOutputBuffer(index,false);
                    if(pcm.n>=limitSeconds*sampleRate) outputDone=true;
                }else if(index==MediaCodec.INFO_OUTPUT_FORMAT_CHANGED){
                    MediaFormat f=codec.getOutputFormat();
                    channels=f.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                    sampleRate=f.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    pcmEncoding=f.containsKey(MediaFormat.KEY_PCM_ENCODING)?f.getInteger(MediaFormat.KEY_PCM_ENCODING):AudioFormat.ENCODING_PCM_16BIT;
                    if(pcmEncoding!=AudioFormat.ENCODING_PCM_16BIT && pcmEncoding!=AudioFormat.ENCODING_PCM_FLOAT)
                        throw new IOException("Decoder PCM encoding unsupported: "+pcmEncoding);
                }else if(++stalls>500) throw new IOException("Audio decoder stalled");
            }
            float[] input=pcm.toArray();
            if(input.length==0) throw new IOException("No decoded audio samples");
            if(sampleRate==RATE) return new AudioData(input);
            int len=(int)Math.round(input.length*RATE/(double)sampleRate);
            float[] resampled=new float[len];
            for(int i=0;i<len;i++){
                double x=i*(sampleRate/(double)RATE);
                int x0=Math.min(input.length-1,(int)x);
                int x1=Math.min(input.length-1,x0+1);
                float a=(float)(x-x0);
                resampled[i]=input[x0]*(1-a)+input[x1]*a;
            }
            return new AudioData(resampled);
        } finally {
            if(codec!=null){ try{codec.stop();}catch(Exception ignored){} codec.release(); }
            extractor.release();
        }
    }

    /** One 5-second chunk, padded to 80000 samples. Output [1,501,512] flattened. */
    static float[] logMel(float[] samples, int start) {
        float[] audio=new float[CHUNK];
        int valid=Math.max(0,Math.min(CHUNK,samples.length-start));
        if(valid>0)System.arraycopy(samples,start,audio,0,valid);
        float[] out=new float[FRAMES*MEL];
        double[] re=new double[FFT],im=new double[FFT];
        double[] hann=new double[FFT];
        for(int j=0;j<FFT;j++) hann[j]=.5-.5*Math.cos(2*Math.PI*j/FFT);
        // HTK frequency bands as in muscriptor.modules.mel_spectrogram (torchaudio norm=None).
        double melMax=2595.0*Math.log10(1.0+(RATE/2.0)/700.0);
        double[] points=new double[MEL+2];
        for(int i=0;i<points.length;i++) points[i]=700*(Math.pow(10.0,melMax*i/((MEL+1.0)*2595.0))-1.0);
        int[] left=new int[FFT/2+1],right=new int[FFT/2+1];
        double[] lw=new double[FFT/2+1],rw=new double[FFT/2+1];
        for(int k=0;k<=FFT/2;k++){
            double hz=k*RATE/(double)FFT;
            int lo=0;while(lo+1<points.length-1&&points[lo+1]<=hz) lo++;
            double t=(hz-points[lo])/(points[lo+1]-points[lo]);
            left[k]=lo-1;right[k]=lo;lw[k]=1-t;rw[k]=t;
        }
        for(int f=0;f<FRAMES;f++){
            int center=f*HOP;
            for(int j=0;j<FFT;j++){
                int p=reflect(center-FFT/2+j,CHUNK);
                re[j]=audio[p]*hann[j];im[j]=0;
            }
            fft(re,im);
            int base=f*MEL;
            for(int k=0;k<=FFT/2;k++){
                double mag=Math.hypot(re[k],im[k]);int a=left[k],b=right[k];
                if(a>=0&&a<MEL)out[base+a]+=mag*lw[k];
                if(b>=0&&b<MEL)out[base+b]+=mag*rw[k];
            }
            for(int j=0;j<MEL;j++) out[base+j]=(float)Math.log(out[base+j]+1e-6);
        }
        return out;
    }
    static int reflect(int i,int length){
        while(i<0||i>=length){if(i<0)i=-i;else i=2*length-2-i;}
        return i;
    }
    static void fft(double[] re,double[] im){
        final int n=re.length;
        for(int i=1,j=0;i<n;i++){
            int bit=n>>1;
            for(; (j&bit)!=0;bit>>=1)j^=bit;
            j^=bit;
            if(i<j){double t=re[i];re[i]=re[j];re[j]=t;t=im[i];im[i]=im[j];im[j]=t;}
        }
        for(int len=2;len<=n;len<<=1){
            double theta=-2*Math.PI/len,wr0=Math.cos(theta),wi0=Math.sin(theta);
            for(int i=0;i<n;i+=len){double wr=1,wi=0;
                for(int j=0;j<len/2;j++){
                    int a=i+j,b=i+j+len/2;
                    double tr=wr*re[b]-wi*im[b],ti=wr*im[b]+wi*re[b];
                    re[b]=re[a]-tr;im[b]=im[a]-ti;re[a]+=tr;im[a]+=ti;
                    double nwr=wr*wr0-wi*wi0;wi=wr*wi0+wi*wr0;wr=nwr;
                }
            }
        }
    }
}
