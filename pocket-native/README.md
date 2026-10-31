# MuScript Pocket — native ONNX Android prototype

This is the original Anima-derived MuScript Pocket UI, wired to a real local
MuScriptor Small INT4 ONNX transcription engine for Android. It includes the
user's already-verified **merged 68.6 MiB ONNX** as an APK asset.

**Important backend limitation:** ONNX Runtime Android has no official Vulkan
execution provider. This project is **native ONNX CPU with optional NNAPI
experimentation**. A true Mali-G720 Vulkan backend requires re-exporting the
PyTorch checkpoint into a different Vulkan-compatible runtime (e.g.
ExecuTorch Vulkan `.pte`) or writing custom Vulkan kernels. Adding a `vulkan`
button to this project would be dishonest. NNAPI may delegate zero INT4
MatMulNBits nodes; use CPU as the baseline.

## Build an APK

### Android Studio

1. Open the **root folder** as an Android Studio project.
2. Let it install **Android SDK 35**, **JDK 17**, and the Gradle Android plugin.
3. Choose **Build > Build APK(s)**.
4. Install `app/build/outputs/apk/debug/app-debug.apk` on your phone.

The model is bundled; no downloads or manually selecting ONNX files on Android.
For best battery/thermal results, start with 5-second audio, 2 threads,
256–512 generated tokens, and the CPU backend.

### GitHub Actions (no local Android SDK)

Upload the **whole extracted project** into a GitHub repository (including the
`app/src/main/assets/model.onnx` file; it is below GitHub's 100 MB file limit).
Go to **Actions > Build MuScript Pocket APK > Run workflow** and download the
`muscript-pocket-debug-apk` artifact from the successful build.

## App

- Original dark Pocket layout with a fixed three-octave piano roll and a
  left-hand keyboard. Drag up/down or tap arrows to shift the displayed range.
- Native Android audio picker and PCM decoding (WAV, MP3, M4A etc. depending
  on the Android platform decoder).
- Stereo-to-mono, resampling to 16 kHz, 2048-point Hann STFT, 512 HTK mel
  magnitude bands, log scaling (`1e-6`).
- 5-second inference chunks, greedy decoding with optional prelude forcing.
- Bundled one-file MuScriptor ONNX: FP32 conditioner + INT4 core.
- MIDI Type 1 export using the Android document picker; cancel button;
  progress and measured end-to-end tokens/second / memory (PSS).
- Inference occurs in a dedicated Java worker, not on the WebView UI thread.
- The native ONNX session stays loaded across multiple transcription runs.

## Model and numerical test information

The original merged ONNX was verified on desktop against the two-file INT4
ONNX pipeline with **exactly zero differences** in logits, predicted tokens,
and KV cache over four steps. That does not certify Android compatibility.

The native audio frontend was tested independently against a NumPy
STFT + HTK-mel reference for a synthetic mixed-tone 5-second clip:
mean absolute difference < 2e-7, max difference < 1.5e-6 in log-mel outputs.
This checks the **mel transform only**. Audio decoding/resampling is native
Android PCM and linear resampling, rather than PyTorch's resampler.

The basic Java token/MIDI reconstruction test passes with a synthetic
note-on and note-off sequence. **The full Android APK and real inference on
Xiaomi 14T Pro have not been tested.** Model loading may fail if Android's
`onnxruntime-android` package lacks a required INT4 operator on ARM64.

## Performance and limitations

- Native ONNX Runtime can be faster than browser WASM but **no speed number is
  promised** until the real Dimensity 9300+ benchmark is collected.
- The current ONNX export returns the **entire increasing FP32 KV cache** each
  decode step, leading to high memory traffic. Fixed / reusable KV cache
  requires a new model export. This can become a bottleneck on long clips.
- Some original MuScriptor decoding features (beam search, native source-code
  boundary-repair heuristics, high-fidelity resampling) are not implemented.
- Multi-chunk prelude forcing is experimental with this ONNX export.
- The MIDI writer approximates native MuScriptor's event handling; compare
  against reference recordings before trusting dense arrangements.
- The three-octave display visualizes completed MIDI notes, not original audio
  spectrum peaks. No synthetic demo notes are generated during inference.
- Audio duration is capped at 180 seconds to prevent unbounded memory use.

## Files

- `app/src/main/assets/model.onnx`: bundled 68.6 MiB merged INT4 checkpoint.
- `app/src/main/assets/index.html`: Pocket UI.
- `MainActivity.java`: native integration + import/export.
- `AudioFrontend.java`: MediaCodec decoding + STFT/HTK mel.
- `OnnxTranscriber.java`: CPU/NNAPI inference and KV cache lifecycle.
- `MidiTracker.java`: token-to-note conversion and MIDI writer.
- `tests/DecoderSelfTest.java`: standalone Java MIDI/token fixture (not bundled in APK).

## Useful links

- https://onnxruntime.ai/docs/tutorials/mobile/
- https://onnxruntime.ai/docs/execution-providers/
- https://docs.pytorch.org/executorch/stable/android-vulkan.html
- https://github.com/sonic-coders/muscriptor
