# MuScriptor ONNX derivative attribution notice

This repository contains derivative ONNX Runtime exports of MuScriptor Medium.

- Original model: https://huggingface.co/MuScriptor/muscriptor-medium
- Original project: https://github.com/muscriptor/muscriptor
- Original developers: Mirelo × Kyutai
- Original authors: Alexandre Rouard, Michael Krause, Axel Roebel,
  Carl-Johann Simon-Gabriel, and Alexandre Défossez
- ONNX conversion and quantization repository: happyme531/muscriptor-medium-onnx

Changes from the original checkpoint:

1. The model was exported to FP32 ONNX conditioner and autoregressive decoder
   graphs with explicit KV-cache inputs and outputs.
2. The decoder attention was rewritten to 24 ONNX Runtime
   `GroupQueryAttention` nodes with past/present shared-buffer support.
3. Forty-eight residual Add + LayerNorm patterns were fused into
   `SkipLayerNormalization` nodes.
4. Ninety-six Transformer-backbone MatMul weights were quantized to symmetric
   INT4 with RTN, group size 32, using ONNX Runtime `MatMulNBits`.
5. Embeddings, the conditioner, and the LM head remain FP32 and unquantized.
   Log-mel audio preprocessing remains outside the ONNX graphs.

The original and derivative model weights are licensed under CC BY-NC 4.0.
Users must also comply with the original model card's supplemental conditions,
including holding all necessary rights to input audio and avoiding illegal or
unauthorized use. See `UPSTREAM_MODEL_CARD.md` for the complete upstream terms,
intended uses, limitations, citation, and disclaimers.

This derivative repository is not affiliated with or endorsed by Mirelo,
Kyutai, or the original authors. The models and generated content are provided
as-is without warranty.
