#!/usr/bin/env python3
"""Create ORT sessions for every generated cache adapter and execute a one-token smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort


def cache_array(type_name: str, shape: tuple[int, ...]) -> np.ndarray:
    if type_name == "tensor(float16)":
        return np.zeros(shape, dtype=np.float16)
    if type_name == "tensor(int8)":
        return np.zeros(shape, dtype=np.int8)
    if type_name == "tensor(uint8)":
        return np.zeros(shape, dtype=np.uint8)
    if type_name == "tensor(float)":
        return np.zeros(shape, dtype=np.float32)
    raise ValueError(f"Unsupported cache input type {type_name}")


def run_variant(path: Path, max_cache: int) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(path.as_posix(), sess_options=options, providers=["CPUExecutionProvider"])

    cache_inputs = [item for item in session.get_inputs() if item.name.startswith("past_")]
    if len(cache_inputs) != 48:
        raise RuntimeError(f"{path.name}: expected 48 cache inputs, found {len(cache_inputs)}")

    cache_type = cache_inputs[0].type
    if cache_type == "tensor(bfloat16)":
        # NumPy has no portable bfloat16 carrier in the ORT wheel. Session creation still validates
        # the graph, types, external weights and kernels; Android's Java API supplies raw uint16.
        print(f"session ok {path.name}: BF16 execution skipped (Python carrier limitation)")
        return

    packed_dim = 32 if cache_type == "tensor(uint8)" else 64
    query_length = 502
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([[1395]], dtype=np.int64),
        "condition_embeddings": np.zeros((1, 501, 1024), dtype=np.float32),
        "position_ids": np.arange(query_length, dtype=np.int64).reshape(1, query_length),
        "seqlens_k": np.array([query_length - 1], dtype=np.int32),
        "total_sequence_length": np.array(query_length, dtype=np.int32),
    }
    for item in cache_inputs:
        feeds[item.name] = cache_array(cache_type, (1, 16, max_cache, packed_dim))

    output_names = ["logits", "present_key.0"]
    logits, present = session.run(output_names, feeds)
    if logits.shape[-1] < 1393:
        raise RuntimeError(f"{path.name}: invalid logits shape {logits.shape}")
    expected_last = packed_dim
    if present.shape[-1] != expected_last:
        raise RuntimeError(f"{path.name}: invalid present shape {present.shape}")
    print(
        f"run ok {path.name}: cache={cache_type}, logits={logits.shape}, "
        f"present={present.shape}/{present.dtype}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, required=True)
    parser.add_argument("--max-cache", type=int, default=512)
    args = parser.parse_args()

    link = args.variants / "decoder.onnx.data"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(args.external_data.resolve(), link)
    try:
        paths = sorted(args.variants.glob("decoder-cache-*.onnx"))
        if not paths:
            raise RuntimeError("No generated decoder cache variants found")
        for path in paths:
            run_variant(path, args.max_cache)
    finally:
        if link.is_symlink():
            link.unlink()


if __name__ == "__main__":
    main()
