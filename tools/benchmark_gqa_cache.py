#!/usr/bin/env python3
"""Benchmark equal-length FP32, native-GQA FP16, and compatibility FP16 cache paths."""
from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

CACHE_NAMES = [
    *(f"past_key.{layer}" for layer in range(24)),
    *(f"past_value.{layer}" for layer in range(24)),
]


def cache_dtype(type_name: str) -> np.dtype:
    if type_name == "tensor(float)":
        return np.dtype(np.float32)
    if type_name == "tensor(float16)":
        return np.dtype(np.float16)
    raise ValueError(f"Unsupported benchmark cache type: {type_name}")


def run_model(label: str, model_path: Path, max_cache: int, steps: int) -> dict[str, float]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path.as_posix(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    cache_inputs = {
        item.name: item
        for item in session.get_inputs()
        if item.name.startswith("past_")
    }
    if len(cache_inputs) != 48:
        raise RuntimeError(f"{label}: expected 48 cache tensors, found {len(cache_inputs)}")
    dtype = cache_dtype(next(iter(cache_inputs.values())).type)
    cache = {
        name: np.zeros((1, 16, max_cache, 64), dtype=dtype)
        for name in cache_inputs
    }
    output_names = [
        "logits",
        *(f"present_key.{layer}" for layer in range(24)),
        *(f"present_value.{layer}" for layer in range(24)),
    ]

    first_times: list[float] = []
    token_times: list[float] = []
    past_length = 0
    token = 1395
    for step in range(steps):
        first = step == 0
        condition_length = 501 if first else 0
        query_length = condition_length + 1
        total_length = past_length + query_length
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.array([[token]], dtype=np.int64),
            "condition_embeddings": np.zeros(
                (1, condition_length, 1024),
                dtype=np.float32,
            ),
            "position_ids": np.arange(
                past_length,
                total_length,
                dtype=np.int64,
            ).reshape(1, query_length),
            "seqlens_k": np.array([total_length - 1], dtype=np.int32),
            "total_sequence_length": np.array(total_length, dtype=np.int32),
            **cache,
        }
        start = time.perf_counter()
        outputs = session.run(output_names, feeds)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if first:
            first_times.append(elapsed_ms)
        else:
            token_times.append(elapsed_ms)

        logits = outputs[0][0]
        token = int(np.argmax(logits[:1393]))
        cache = {
            f"past_key.{layer}": outputs[1 + layer]
            for layer in range(24)
        }
        cache.update({
            f"past_value.{layer}": outputs[1 + 24 + layer]
            for layer in range(24)
        })
        past_length = total_length

    steady = statistics.median(token_times) if token_times else float("nan")
    print(
        f"{label}: first={first_times[0]:.2f} ms, "
        f"steady_median={steady:.2f} ms/token, steps={steps}, cache={max_cache}"
    )
    return {"first": first_times[0], "steady": steady}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, required=True)
    parser.add_argument("--max-cache", type=int, default=768)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="muscriptor-cache-bench-") as temp:
        root = Path(temp)
        shutil.copyfile(args.external_data, root / "decoder.onnx.data")
        base = root / "decoder-fp32.onnx"
        native = root / "decoder-fp16-native.onnx"
        compat = root / "decoder-fp16-compat.onnx"
        shutil.copyfile(args.decoder, base)
        shutil.copyfile(args.variants / "decoder-cache-fp16-native-gqa.onnx", native)
        shutil.copyfile(args.variants / "decoder-cache-fp16.onnx", compat)

        results = {
            "FP32": run_model("FP32", base, args.max_cache, args.steps),
            "FP16 native GQA": run_model(
                "FP16 native GQA",
                native,
                args.max_cache,
                args.steps,
            ),
            "FP16 compatibility": run_model(
                "FP16 compatibility",
                compat,
                args.max_cache,
                args.steps,
            ),
        }
        fp32 = results["FP32"]["steady"]
        print(
            "ratios vs FP32 steady: "
            f"native={results['FP16 native GQA']['steady'] / fp32:.3f}x, "
            f"compat={results['FP16 compatibility']['steady'] / fp32:.3f}x"
        )


if __name__ == "__main__":
    main()
