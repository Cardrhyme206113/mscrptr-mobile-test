#!/usr/bin/env python3
"""Execute the custom K8/V8 graph and compare incremental logits with native FP16 GQA."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

LAYERS = 24
HEADS = 16
HEAD_DIM = 64
VOCAB_LIMIT = 1393
INITIAL_TOKEN = 1395
CONDITION_LENGTH = 501


def present_names() -> list[str]:
    names: list[str] = []
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            names.append(f"present_{kind}.{layer}")
    return names


def scale_present_names() -> list[str]:
    names: list[str] = []
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            names.append(f"present_{kind}_scale.{layer}")
    return names


def fp16_cache(max_cache: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            result[f"past_{kind}.{layer}"] = np.zeros(
                (1, HEADS, max_cache, HEAD_DIM),
                dtype=np.float16,
            )
    return result


def quantize_cache(
    source: dict[str, np.ndarray],
    max_cache: int,
    positions: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for name, source_array in source.items():
        active = source_array[:, :, :positions, :].astype(np.float32)
        scale = np.maximum(np.max(np.abs(active), axis=-1), 1.0e-6) / 127.0
        quantized = np.clip(np.rint(active / scale[..., None]), -127, 127).astype(np.int8)
        destination = np.zeros((1, HEADS, max_cache, HEAD_DIM), dtype=np.int8)
        scale_destination = np.zeros((1, HEADS, max_cache), dtype=np.float32)
        destination[:, :, :positions, :] = quantized
        scale_destination[:, :, :positions] = scale
        data[name] = destination
        scales[name.replace("past_", "past_") + "_unused"] = scale_destination

    corrected_scales: dict[str, np.ndarray] = {}
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            corrected_scales[f"past_{kind}_scale.{layer}"] = scales[
                f"past_{kind}.{layer}_unused"
            ]
    return data, corrected_scales


def run_fp16_prefill(
    session: ort.InferenceSession,
    max_cache: int,
) -> tuple[int, dict[str, np.ndarray], np.ndarray]:
    cache = fp16_cache(max_cache)
    total = CONDITION_LENGTH + 1
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([[INITIAL_TOKEN]], dtype=np.int64),
        "condition_embeddings": np.zeros((1, CONDITION_LENGTH, 1024), dtype=np.float32),
        "position_ids": np.arange(total, dtype=np.int64).reshape(1, total),
        "seqlens_k": np.array([total - 1], dtype=np.int32),
        "total_sequence_length": np.array(total, dtype=np.int32),
        **cache,
    }
    outputs = session.run(["logits", *present_names()], feeds)
    logits = np.asarray(outputs[0]).reshape(-1)
    token = int(np.argmax(logits[:VOCAB_LIMIT]))
    result_cache: dict[str, np.ndarray] = {}
    cursor = 1
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            result_cache[f"past_{kind}.{layer}"] = outputs[cursor]
            cursor += 1
    return token, result_cache, logits


def run_fp16_incremental(
    session: ort.InferenceSession,
    token: int,
    past_length: int,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    total = past_length + 1
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([[token]], dtype=np.int64),
        "condition_embeddings": np.zeros((1, 0, 1024), dtype=np.float32),
        "position_ids": np.array([[past_length]], dtype=np.int64),
        "seqlens_k": np.array([total - 1], dtype=np.int32),
        "total_sequence_length": np.array(total, dtype=np.int32),
        **cache,
    }
    started = time.perf_counter()
    outputs = session.run(["logits", *present_names()], feeds)
    elapsed = (time.perf_counter() - started) * 1000.0
    result_cache: dict[str, np.ndarray] = {}
    cursor = 1
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            result_cache[f"past_{kind}.{layer}"] = outputs[cursor]
            cursor += 1
    return np.asarray(outputs[0]).reshape(-1), result_cache, elapsed


def run_k8v8_incremental(
    session: ort.InferenceSession,
    token: int,
    past_length: int,
    data: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], float]:
    total = past_length + 1
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([[token]], dtype=np.int64),
        "condition_embeddings": np.zeros((1, 0, 1024), dtype=np.float32),
        "position_ids": np.array([[past_length]], dtype=np.int64),
        "seqlens_k": np.array([total - 1], dtype=np.int32),
        "total_sequence_length": np.array(total, dtype=np.int32),
        **data,
        **scales,
    }
    names = ["logits", *present_names(), *scale_present_names()]
    started = time.perf_counter()
    outputs = session.run(names, feeds)
    elapsed = (time.perf_counter() - started) * 1000.0

    next_data: dict[str, np.ndarray] = {}
    next_scales: dict[str, np.ndarray] = {}
    cursor = 1
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            next_data[f"past_{kind}.{layer}"] = outputs[cursor]
            cursor += 1
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            next_scales[f"past_{kind}_scale.{layer}"] = outputs[cursor]
            cursor += 1
    return np.asarray(outputs[0]).reshape(-1), next_data, next_scales, elapsed


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left[:VOCAB_LIMIT].astype(np.float64)
    right = right[:VOCAB_LIMIT].astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(denominator, 1.0e-30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--k8v8", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, required=True)
    parser.add_argument("--custom-library", type=Path, required=True)
    parser.add_argument("--max-cache", type=int, default=768)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="muscriptor-k8v8-") as temp:
        root = Path(temp)
        shutil.copyfile(args.external_data, root / "decoder.onnx.data")
        fp16_path = root / "decoder-fp16.onnx"
        k8v8_path = root / "decoder-k8v8.onnx"
        shutil.copyfile(args.fp16, fp16_path)
        shutil.copyfile(args.k8v8, k8v8_path)

        fp16_options = ort.SessionOptions()
        fp16_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        fp16_options.intra_op_num_threads = 2
        fp16_options.inter_op_num_threads = 1
        fp16 = ort.InferenceSession(
            fp16_path.as_posix(),
            sess_options=fp16_options,
            providers=["CPUExecutionProvider"],
        )

        k8_options = ort.SessionOptions()
        k8_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        k8_options.intra_op_num_threads = 2
        k8_options.inter_op_num_threads = 1
        k8_options.register_custom_ops_library(args.custom_library.as_posix())
        k8 = ort.InferenceSession(
            k8v8_path.as_posix(),
            sess_options=k8_options,
            providers=["CPUExecutionProvider"],
        )

        token, fp16_state, _ = run_fp16_prefill(fp16, args.max_cache)
        k8_data, k8_scales = quantize_cache(
            fp16_state,
            args.max_cache,
            CONDITION_LENGTH + 1,
        )
        agreements = 0
        cosines: list[float] = []
        fp16_times: list[float] = []
        k8_times: list[float] = []
        past = CONDITION_LENGTH + 1

        for step in range(args.steps):
            fp16_logits, fp16_state, fp16_ms = run_fp16_incremental(
                fp16,
                token,
                past,
                fp16_state,
            )
            k8_logits, k8_data, k8_scales, k8_ms = run_k8v8_incremental(
                k8,
                token,
                past,
                k8_data,
                k8_scales,
            )
            fp16_token = int(np.argmax(fp16_logits[:VOCAB_LIMIT]))
            k8_token = int(np.argmax(k8_logits[:VOCAB_LIMIT]))
            similarity = cosine(fp16_logits, k8_logits)
            agreements += int(fp16_token == k8_token)
            cosines.append(similarity)
            fp16_times.append(fp16_ms)
            k8_times.append(k8_ms)
            print(
                f"step {step + 1}: fp16={fp16_token}, k8v8={k8_token}, "
                f"cosine={similarity:.6f}, host_ms={fp16_ms:.2f}/{k8_ms:.2f}"
            )
            if not np.all(np.isfinite(k8_logits[:VOCAB_LIMIT])):
                raise RuntimeError("K8/V8 produced non-finite logits")
            token = fp16_token
            past += 1

        minimum_cosine = min(cosines)
        print(
            f"summary: top1_agreement={agreements}/{args.steps}, "
            f"minimum_cosine={minimum_cosine:.6f}, "
            f"host_allocating_output_ms={np.median(fp16_times):.2f}/{np.median(k8_times):.2f}"
        )
        if minimum_cosine < 0.90:
            raise RuntimeError(f"K8/V8 logit cosine too low: {minimum_cosine:.6f}")


if __name__ == "__main__":
    main()
