---
license: cc-by-nc-4.0
library_name: onnxruntime
base_model: MuScriptor/muscriptor-medium
tags:
  - muscriptor
  - music-transcription
  - automatic-music-transcription
  - audio-to-midi
  - onnx
  - onnxruntime
  - int4
---

# MuScriptor Medium — optimized W4A32 ONNX Runtime

This is an **unofficial derivative ONNX conversion** of
[`MuScriptor/muscriptor-medium`](https://huggingface.co/MuScriptor/muscriptor-medium),
developed by Mirelo × Kyutai. It is not affiliated with or endorsed by the
MuScriptor authors.

The conversion uses upstream revision
`f32236969308476e01fd3aae67357de5feb05a2d`. The source checkpoint SHA-256 is
`ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec`.

## Model variant

The deployable model is in `onnx/w4a32_optimized`.

| Component | Storage / graph dtype |
|---|---|
| Transformer backbone | symmetric RTN INT4, group size 32 |
| Graph activations and KV cache | FP32 |
| Token and metadata embeddings | FP32, not quantized |
| LM head | FP32, not quantized |
| Conditioner | FP32, not quantized |

Only the 96 constant-weight MatMul operations in the 24-layer Transformer
backbone are converted to ONNX Runtime `MatMulNBits`. The graph uses
`accuracy_level=4`: its public tensors remain FP32, while supported CPU kernels
dynamically quantize MatMul activations to INT8 and accumulate in INT32.

The decoder additionally contains 24 `GroupQueryAttention` and 48
`SkipLayerNormalization` nodes. Its fixed-capacity KV cache supports ONNX
Runtime past/present buffer sharing through I/O Binding, avoiding allocation
and copying of the complete cache on every token.

`decoder.onnx.data` contains external tensor data and **must remain next to**
`decoder.onnx`.

## Quick smoke test

```bash
hf download happyme531/muscriptor-medium-onnx --local-dir muscriptor-medium-onnx
cd muscriptor-medium-onnx
uv venv
source .venv/bin/activate
uv pip install -r runtime/requirements.txt
python runtime/test_onnxruntime.py \
  --model-dir onnx/w4a32_optimized \
  --provider cpu --max-new-tokens 8 --intra-op-threads 16
```

Use `--audio path/to/audio.wav` to replace the deterministic five-second
440 Hz smoke-test tone. The helper validates low-level conditioner/decoder
execution and greedy token generation; it is not the complete upstream
multi-chunk audio-to-MIDI application.

The default cache length is 2504 positions. A shared FP32 cache at that length
occupies about 469.5 MiB. Lower `--max-cache-length` when the deployment has a
smaller generation budget.

## Validation

The model passes `onnx.checker.check_model` and was executed with ONNX Runtime
1.26.0 on x86-64 CPU. Results below use a Xeon Gold 6278C, one NUMA node,
16 intra-op threads, shared KV buffers, a five-second 440 Hz input, and a
576-position cache:

| Metric | Result |
|---|---:|
| Prefill | 468.0 ms |
| Steady decode, mean over 56 measured tokens | 7.74 ms/token |
| Steady decode median | 7.73 ms/token |
| Steady decode p90 | 7.85 ms/token |

A 64-step teacher-forced comparison against the original FP32 PyTorch model
gave 81.25% top-1 agreement, 0.741 mean top-5 overlap, 0.286 centered relative
logit L2 error, and 0.208 mean probability total variation. This synthetic
smoke input is useful for regression testing, not a transcription-quality
benchmark.

ARM64 execution was not measured on the target SoC. ONNX Runtime's CPU kernels
provide an ARM64 path for this graph: the quantized backbone can use MLAS NEON
dot-product kernels, while GQA's FP32 matrix multiplications use MLAS SGEMM.
KleidiAI/SME acceleration additionally depends on the ONNX Runtime build and
the device's runtime CPU features.

## Files

| File | Size | SHA-256 |
|---|---:|---|
| `onnx/w4a32_optimized/conditioner.onnx` | 6,223,112 bytes | `c1ea7895ade9760538f42f63110a782bcc1606dee89f2ee434db2251434a3c7d` |
| `onnx/w4a32_optimized/decoder.onnx` | 79,814 bytes | `74e1181288fa5205754e9661cbccafc5182d035beadd4412da1ecd00515c3de9` |
| `onnx/w4a32_optimized/decoder.onnx.data` | 217,354,240 bytes | `4bc6ac2807854632bacc0ef9b5e9e1871545e573194fe7fc197948730aa3c8ef` |

Total model storage, excluding the small JSON config, is 223,657,166 bytes
(213.30 MiB).

## License, attribution, and conditions

The original model weights and these derivative ONNX weights are distributed
under [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE).
Commercial use is not permitted under that license. Users must also comply
with the upstream supplemental conditions, including holding all necessary
rights to input audio and complying with applicable laws.

See [NOTICE.md](NOTICE.md) for attribution and
[UPSTREAM_MODEL_CARD.md](UPSTREAM_MODEL_CARD.md) for the original model card,
intended uses, limitations, supplemental conditions, author list, and citation.
The runtime helpers are covered by [their MIT license](runtime/LICENSE).
