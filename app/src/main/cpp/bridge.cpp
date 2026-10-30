#include <jni.h>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <muscriptor/muscriptor.hpp>

static std::atomic<bool> cancelled{false};
static std::string errorString(const std::string& s){return s;}
static std::string noteJson(const std::vector<msl::Note>& ns) {
    std::ostringstream o; o << '[';o << std::fixed << std::setprecision(3);
    bool first=true;
    for(const auto& n:ns) {
        if(n.pitch<0 || n.pitch>127 || n.offset<=n.onset)continue;
        if(!first)o<<',';first=false;
        o<<'['<<n.pitch<<','<<n.onset<<','<<(n.offset-n.onset)<<']';
    }
    o<<']'; return o.str();
}
static void report(JNIEnv* env,jobject object,jmethodID callback,const std::vector<msl::Note>& notes,float progress,double sec){
    const std::string json=noteJson(notes);
    jstring value=env->NewStringUTF(json.c_str());
    if(value){env->CallVoidMethod(object,callback,value,progress,sec);env->DeleteLocalRef(value);}
}
static void fail(JNIEnv* env,const std::string& message){
    jclass cls=env->FindClass("java/lang/IllegalStateException");
    if(cls){env->ThrowNew(cls,message.c_str());env->DeleteLocalRef(cls);}
}

// Re-transcribe with a 2.5 s offset, so problematic 5 s boundaries land in
// the *middle* of chunks. Replace events with onsets close to 5 s boundaries.
// This is deliberately experimental. Two passes cost roughly twice the inference.
static std::vector<msl::Note> repairBoundaries(
    const std::vector<msl::Note>& first, const std::vector<msl::Note>& shifted, double duration) {
    constexpr double width=.7, shift=2.5;
    auto nearBoundary=[](double t)->bool{
        if(t<=4.0)return false;
        const double b=std::round(t/5.0)*5.0;
        return std::abs(b-t)<=width;
    };
    std::vector<msl::Note> merged;
    for(const auto& n:first) {
        // Preserve notes that start away from a boundary.
        if(!nearBoundary(n.onset))merged.push_back(n);
    }
    for(auto n:shifted){
        n.onset-=shift; n.offset-=shift;
        if(n.onset<0 || n.onset>duration || n.offset<=0 || !nearBoundary(n.onset))continue;
        n.onset=std::max(0.0,n.onset);n.offset=std::min(duration,n.offset);
        if(n.offset>n.onset)merged.push_back(n);
    }
    std::sort(merged.begin(),merged.end(),[](const auto&a,const auto&b){
        if(a.onset!=b.onset)return a.onset<b.onset;
        if(a.pitch!=b.pitch)return a.pitch<b.pitch;
        return a.program<b.program;
    });
    // Remove near-identical retriggers, while retaining deliberate repetitions.
    std::vector<msl::Note> output;
    for(auto n:merged){
        bool duplicate=false;
        for(auto it=output.rbegin();it!=output.rend();++it){
            if(it->onset<n.onset-.075)break;
            if(it->pitch==n.pitch && it->program==n.program && std::abs(it->onset-n.onset)<.045){
                it->offset=std::max(it->offset,n.offset);duplicate=true;break;
            }
        }
        if(!duplicate)output.push_back(n);
    }
    return output;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_muscriptor_app_MainActivity_nativeTranscribe(JNIEnv* env, jobject object,
    jstring modelPath, jfloatArray inputSamples, jboolean prelude, jboolean pianoOnly, jboolean repair) {
    cancelled.store(false);
    if(!modelPath || !inputSamples) { fail(env,"Missing model or samples");return nullptr; }
    const char* path=env->GetStringUTFChars(modelPath,nullptr);
    if(!path){fail(env,"Invalid model path");return nullptr;}
    std::string model(path);env->ReleaseStringUTFChars(modelPath,path);
    const jsize count=env->GetArrayLength(inputSamples);
    std::vector<float> samples(static_cast<size_t>(count));
    env->GetFloatArrayRegion(inputSamples,0,count,samples.data());
    if(env->ExceptionCheck())return nullptr;
    jclass cls=env->GetObjectClass(object);
    jmethodID onProgress=env->GetMethodID(cls,"onNativeProgress","(Ljava/lang/String;FD)V");
    env->DeleteLocalRef(cls);
    if(!onProgress){fail(env,"Progress callback missing");return nullptr;}
    try {
        msl::LoadOptions load;
        load.should_cancel=[](){return cancelled.load();};
        auto transcriber=msl::Transcriber::load(model,load);
        if(!transcriber)throw std::runtime_error("Model load: "+std::string(msl::describe(transcriber.error())));
        msl::TranscribeOptions options;
        options.prelude_forcing=prelude;
        if(pianoOnly)options.instruments={msl::InstrumentGroup::AcousticPiano};
        options.should_cancel=[](){return cancelled.load();};
        float firstPart=repair?.5f:1.f;
        auto first=transcriber->transcribe(samples,options,[&](const msl::TranscriptionUpdate& u){
            std::vector<msl::Note> batch(u.new_notes.begin(),u.new_notes.end());
            report(env,object,onProgress,batch,u.progress*firstPart,u.finalized_through);
            return !cancelled.load()&&!env->ExceptionCheck();
        });
        if(!first)throw std::runtime_error("Inference: "+std::string(msl::describe(first.error())));
        std::vector<msl::Note> result=std::move(*first);
        if(repair && !cancelled.load()) {
            std::vector<float> alternate(40000,0.f);
            alternate.insert(alternate.end(),samples.begin(),samples.end());
            auto second=transcriber->transcribe(alternate,options,[&](const msl::TranscriptionUpdate& u){
                report(env,object,onProgress,{},.5f+u.progress*.5f,std::max(0.0,u.finalized_through-2.5));
                return !cancelled.load()&&!env->ExceptionCheck();
            });
            if(!second)throw std::runtime_error("Repair pass: "+std::string(msl::describe(second.error())));
            result=repairBoundaries(result,*second,samples.size()/16000.0);
        }
        std::string json=noteJson(result);
        return env->NewStringUTF(json.c_str());
    } catch(const std::exception& e) {fail(env,e.what());return nullptr;}
}
extern "C" JNIEXPORT void JNICALL
Java_io_muscriptor_app_MainActivity_nativeCancel(JNIEnv*,jobject){cancelled.store(true);}
