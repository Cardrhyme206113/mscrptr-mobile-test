# MuScriptor Mobile

Experimental Android ARM64 app for running the unofficial INT4 ONNX conversion of MuScriptor Medium fully on-device.

## Current features

- Anonymous first-run model download from `happyme531/muscriptor-medium-onnx` (no Hugging Face token or account required)
- SHA-256 verification and resumable downloads
- Android audio picker and MediaCodec decoding for common phone audio formats
- 16 kHz mono preprocessing and MuScriptor-compatible 501×512 log-mel frontend
- ONNX Runtime CPU inference with fused GQA and shared pinned KV-cache buffers
- Five-second chunking with cross-chunk tie prelude forcing
- Token-by-token note streaming into a live scrolling piano roll
- Standard MIDI file export
- ARM64-only APK, targeted at the Xiaomi 14T Pro / Dimensity 9300+ class of devices

## Model footprint

The downloaded model is 223,657,166 bytes (213.30 MiB). The FP32 KV cache is the larger runtime allocation:

- Low memory: 1024 positions, about 192 MiB
- Balanced: 1536 positions, about 288 MiB
- Full: 2504 positions, about 469.5 MiB

The app downloads these public files directly:

- `conditioner.onnx`
- `decoder.onnx`
- `decoder.onnx.data`

`decoder.onnx.data` must remain beside `decoder.onnx`.

## Build

GitHub Actions builds a debug APK on every relevant push. Open **Actions → Build Android APK**, select the latest successful run, and download the `muscriptor-mobile-debug-apk` artifact.

Local build with Java 17 and Gradle 8.11.1:

```bash
gradle :app:assembleDebug
```

The APK will be at `app/build/outputs/apk/debug/app-debug.apk`.

## Important meaning of “streaming”

The piano roll and note count update while autoregressive inference is still running. This does not promise that inference itself is faster than wall-clock audio duration; actual speed must be measured on the target phone.

## Status / limitations

This is an early device test build. CPU inference is implemented first because the model uses custom/contrib ONNX Runtime ops (`MatMulNBits`, `GroupQueryAttention`, and `SkipLayerNormalization`) that are not expected to map cleanly to Android NNAPI. The app currently uses greedy decoding, fixed 120 BPM MIDI timing, and no beat-grid detection. Dense chunks can hit a reduced cache profile's token limit; the full 2504-position profile matches the exported model's recommended cache.

## License

The derivative model weights are CC BY-NC 4.0 and are not for commercial use. The user must have the necessary rights to any audio supplied to the model. See `docs/model-reference/NOTICE.md` and the upstream model card material in `docs/model-reference`.
