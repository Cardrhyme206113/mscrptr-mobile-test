# MuScriptor Android — SVG front end + native inference

An Android WebView application using the full-width, full MIDI 0–127 SVG piano-roll UI from this conversation. The view is offline, uses Web Audio for MIDI playback, and renders note updates sent by an Android JNI bridge backed by `DamRsn/muscriptor.cpp` (GGUF, FP16 and Vulkan).

## Build an APK with GitHub Actions

1. This clean project is on the `vulkan-small-clean` branch; the old ONNX branch is not modified.
2. Open **Actions → Android APK**, select the `vulkan-small-clean` branch, and run the workflow (pushes on this branch also run it).
3. Download the `MuScriptor-Android-APK` workflow artifact and extract `app-debug.apk`.
4. Install that APK on an ARM64 Android device. Allow installation from that source if prompted.

The workflow downloads the Android NDK and builds the native backend, then assembles a debug-signed APK. **The first native Vulkan build can take several minutes or more; it is not guaranteed to finish under one minute.** This Android/Vulkan integration has *not* been hardware-tested yet. If the build fails, use the workflow logs to identify the compiler/SDK issue. APK must be built by CI/Android Studio; this source ZIP is not itself installable.

### Loading a model

Model weights are not bundled. Download `v1/muscriptor-small-f16.gguf` from [DamRsn/muscriptor-gguf](https://huggingface.co/DamRsn/muscriptor-gguf), open the APK, tap **Settings → Model → Choose**, and select the GGUF through Android's file picker. The app copies the model into its private storage and reuses it on subsequent launches. Select an MP3 or WAV and tap **Transcribe**. Notes arrive from the native model after each 5-second chunk, and final output can be saved using Android's document picker.

### Settings

- **Boundary repair** (experimental): runs a *second transcription* with 2.5 seconds of initial silence to shift the 5-second boundaries, and uses notes from the shifted pass close to the original boundaries. This roughly doubles inference cost and may change accuracy. Not a proven fix.
- **Prelude forcing**: the C++ model's built-in note-continuity option.
- **Piano only**: true `InstrumentGroup::AcousticPiano` conditioning.
- **KV cache**: internal, automatic.
- **Beam search**: not exposed, because the C++ implementation uses greedy decoding. The HTML demo's fake beam option is hidden in the APK.

### Important limitations

- Streaming emits notes **per processed 5-second chunk**, with notes possibly delayed by one chunk, not instantaneous note-by-note inference. This comes from the C++ model API.
- MIDI output exports pitch, onset and note duration at fixed velocity 87. The model **does not predict velocities**. The current simple MIDI writer flattens different instruments to one piano track; program assignments are not preserved yet.
- The front end remains independently usable without model weights as a MIDI stress test, player and exporter.
- The device, build workflow, and accuracy of Vulkan versus CPU have not been verified on a physical Mali-G720 device.
- The native C++ package can make the APK substantially larger than the HTML prototype; measure the actual artifact.
- Inference accepts MP3/WAV through Android MediaExtractor/MediaCodec and windowed-sinc downsampling to 16 kHz. This is native decoding, not an Internet service.

## Local build

Requirements: Android SDK API 35, NDK r27c, CMake ≥3.24, Ninja, a Vulkan shader compiler (`glslc`), Gradle 8.9, JDK 17.

The workflow `.github/workflows/android-apk.yml` provides the exact native CMake and Gradle commands. Copy the compiled `libmuscriptor_jni.so` into `app/src/main/jniLibs/arm64-v8a/` before running `gradle :app:assembleDebug`.

External source: https://github.com/DamRsn/muscriptor.cpp (GGUF native port). See its license and linked `ggml` notices before redistribution.
