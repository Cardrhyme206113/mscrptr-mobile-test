#!/usr/bin/env python3
"""Run cached autoregressive MuScriptor inference with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
# Importing PyTorch first preloads the CUDA/cuDNN shared libraries shipped in
# the uv environment when available; CPU-only users do not need PyTorch.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
import onnxruntime as ort

from muscriptor_onnx.audio import SAMPLE_RATE, load_audio_16k, log_mel_spectrogram


def device_value(
    array: np.ndarray,
    device_type: str,
    device_id: int,
) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_numpy(array, device_type, device_id)


def run_shared_gqa(
    decoder: ort.InferenceSession,
    feeds: dict[str, ort.OrtValue],
    caches: dict[str, ort.OrtValue],
    layers: int,
    device_type: str,
    device_id: int,
) -> np.ndarray:
    """Run fused GQA while aliasing every present cache to its past cache."""
    binding = decoder.io_binding()
    for name, value in feeds.items():
        binding.bind_ortvalue_input(name, value)
    for name, value in caches.items():
        binding.bind_ortvalue_input(name, value)
    binding.bind_output("logits", device_type, device_id)
    for layer_index in range(layers):
        binding.bind_ortvalue_output(
            f"present_key.{layer_index}", caches[f"past_key.{layer_index}"]
        )
        binding.bind_ortvalue_output(
            f"present_value.{layer_index}", caches[f"past_value.{layer_index}"]
        )
    decoder.run_with_iobinding(binding)
    return binding.get_outputs()[0].numpy()


def providers(requested: str, device_id: int) -> list:
    available = ort.get_available_providers()
    if requested == "cuda" or (requested == "auto" and "CUDAExecutionProvider" in available):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDA EP unavailable; installed providers: {available}")
        return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def session(
    path: Path,
    selected_providers: list,
    profile_prefix: str | None = None,
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        options.inter_op_num_threads = inter_op_threads
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    result = ort.InferenceSession(path, sess_options=options, providers=selected_providers)
    requested_cuda = bool(selected_providers) and (
        selected_providers[0] == "CUDAExecutionProvider"
        or (
            isinstance(selected_providers[0], tuple)
            and selected_providers[0][0] == "CUDAExecutionProvider"
        )
    )
    if requested_cuda and "CUDAExecutionProvider" not in result.get_providers():
        raise RuntimeError("CUDA EP was requested but session creation fell back to CPU")
    return result


def activation_numpy_dtype(metadata: dict) -> np.dtype:
    dtype_name = metadata.get("activation_dtype", metadata.get("dtype"))
    try:
        return {
            "float32": np.dtype(np.float32),
            "float16": np.dtype(np.float16),
        }[dtype_name]
    except KeyError as error:
        raise ValueError(f"unsupported activation dtype: {dtype_name!r}") from error


def causal_mask(
    query_length: int,
    past_length: int,
    dtype: np.dtype = np.dtype(np.float16),
) -> np.ndarray:
    query_positions = past_length + np.arange(query_length)[:, None]
    key_positions = np.arange(past_length + query_length)[None, :]
    masked_value = -65504.0 if dtype == np.dtype(np.float16) else -1.0e9
    return np.where(key_positions <= query_positions, 0.0, masked_value).astype(dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-cache-length", type=int, default=2504)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument(
        "--instrument-id",
        type=int,
        action="append",
        help="Optional MT3_FULL_PLUS group ID; repeat for multiple groups",
    )
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / "config.json").read_text())
    activation_dtype = activation_numpy_dtype(metadata)
    selected = providers(args.provider, args.device_id)
    conditioner = session(
        args.model_dir / "conditioner.onnx",
        selected,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
    )
    decoder = session(
        args.model_dir / "decoder.onnx",
        selected,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
    )

    if args.audio:
        audio = load_audio_16k(args.audio)
        source = str(args.audio)
    else:
        # Deterministic smoke-test input; it only tests execution, not quality.
        time_axis = np.arange(5 * SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        audio = (0.1 * np.sin(2 * np.pi * 440.0 * time_axis)).astype(np.float32)
        source = "generated 440 Hz sine"
    mel = log_mel_spectrogram(audio).astype(activation_dtype)
    instrument_ids = np.asarray(
        [args.instrument_id if args.instrument_id is not None else [-1]], dtype=np.int64
    )
    dataset_ids = np.asarray([[-1]], dtype=np.int64)

    start = time.perf_counter()
    condition = conditioner.run(
        ["condition_embeddings"],
        {
            "log_mel": mel,
            "instrument_ids": instrument_ids,
            "dataset_ids": dataset_ids,
        },
    )[0]
    condition_seconds = time.perf_counter() - start

    layers = metadata["num_layers"]
    heads = metadata["num_heads"]
    head_dim = metadata["head_dim"]
    fused_gqa = "past_key.0" in {value.name for value in decoder.get_inputs()}
    active = decoder.get_providers()
    device_type = "cuda" if active[0] == "CUDAExecutionProvider" else "cpu"
    if fused_gqa:
        cache_bytes = (
            layers
            * 2
            * heads
            * args.max_cache_length
            * head_dim
            * activation_dtype.itemsize
        )
        caches = {
            f"past_{kind}.{layer}": device_value(
                np.zeros(
                    (1, heads, args.max_cache_length, head_dim),
                    dtype=activation_dtype,
                ),
                device_type,
                args.device_id,
            )
            for layer in range(layers)
            for kind in ("key", "value")
        }
    else:
        past_key = np.zeros(
            (layers, 1, heads, 0, head_dim), dtype=activation_dtype
        )
        past_value = np.zeros_like(past_key)
    input_ids = np.asarray([[metadata["initial_token_id"]]], dtype=np.int64)
    generated: list[int] = []
    decode_times: list[float] = []

    for step in range(args.max_new_tokens):
        prefix = condition if step == 0 else np.empty(
            (1, 0, heads * head_dim), activation_dtype
        )
        query_length = prefix.shape[1] + input_ids.shape[1]
        tick = time.perf_counter()
        if fused_gqa:
            past_length = 0 if step == 0 else condition.shape[1] + step
            total_length = past_length + query_length
            if total_length > args.max_cache_length:
                raise ValueError(
                    f"sequence length {total_length} exceeds --max-cache-length "
                    f"{args.max_cache_length}"
                )
            feeds = {
                "input_ids": device_value(input_ids, device_type, args.device_id),
                "condition_embeddings": device_value(prefix, device_type, args.device_id),
                "position_ids": device_value(
                    np.arange(past_length, total_length, dtype=np.int64)[None, :],
                    device_type,
                    args.device_id,
                ),
                "seqlens_k": device_value(
                    np.asarray([total_length - 1], dtype=np.int32),
                    device_type,
                    args.device_id,
                ),
                "total_sequence_length": device_value(
                    np.asarray(total_length, dtype=np.int32),
                    device_type,
                    args.device_id,
                ),
            }
            logits = run_shared_gqa(
                decoder, feeds, caches, layers, device_type, args.device_id
            )
        else:
            mask = causal_mask(query_length, past_key.shape[3], activation_dtype)
            logits, past_key, past_value = decoder.run(
                ("logits", "present_key", "present_value"),
                {
                    "input_ids": input_ids,
                    "condition_embeddings": prefix,
                    "past_key": past_key,
                    "past_value": past_value,
                    "attention_mask": mask,
                },
            )
        decode_times.append(time.perf_counter() - tick)
        if not np.isfinite(logits).all():
            raise RuntimeError("decoder produced non-finite logits")
        logits[:, metadata["first_reserved_token_id"] :] = -np.inf
        token = int(logits.argmax(axis=-1)[0])
        generated.append(token)
        input_ids = np.asarray([[token]], dtype=np.int64)
        if token == metadata["eos_token_id"]:
            break

    sizes = {
        path.name: path.stat().st_size
        for path in args.model_dir.iterdir()
        if path.is_file() and (path.suffix == ".onnx" or path.name.endswith(".onnx.data"))
    }
    sizes["total"] = sum(sizes.values())
    print(f"model:             {args.model_dir}")
    print(f"audio:             {source}")
    print(f"providers:         {active}")
    print(f"mel shape:         {mel.shape}")
    print(f"condition shape:   {condition.shape}")
    if fused_gqa:
        print(
            "shared KV cache:   "
            f"{args.max_cache_length} positions, {cache_bytes / 2**20:.1f} MiB"
        )
    else:
        print(f"final KV shape:    {past_key.shape}")
    print(f"generated tokens:  {generated}")
    print(f"condition latency: {condition_seconds * 1000:.1f} ms")
    print(f"decode latency:    {[round(x * 1000, 1) for x in decode_times]} ms")
    print(f"ONNX file sizes:   {sizes}")


if __name__ == "__main__":
    main()
