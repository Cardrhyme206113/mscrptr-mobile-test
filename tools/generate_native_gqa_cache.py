#!/usr/bin/env python3
"""Generate a native FP16 KV-cache decoder using fused GroupQueryAttention.

Unlike the compatibility cache adapter, this rewrite does not cast the full cache to FP32 on every
layer and token. Each fused GroupQueryAttention node receives FP16 QKV, reads/writes FP16 past and
present tensors directly, and casts only its comparatively tiny attention activation back to FP32
for the surrounding W4A32 decoder graph.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import onnx
from onnx import TensorProto, helper

CACHE_INPUT = re.compile(r"^past_(?:key|value)\.\d+$")
CACHE_OUTPUT = re.compile(r"^present_(?:key|value)\.\d+$")
GQA_DOMAIN = "com.microsoft"
GQA_OP = "GroupQueryAttention"


def clean(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def set_elem_type(value_info: onnx.ValueInfoProto, elem_type: int) -> None:
    value_info.type.tensor_type.elem_type = elem_type


def convert(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    graph = model.graph

    cache_inputs = [value for value in graph.input if CACHE_INPUT.match(value.name)]
    cache_outputs = [value for value in graph.output if CACHE_OUTPUT.match(value.name)]
    if len(cache_inputs) != 48 or len(cache_outputs) != 48:
        raise ValueError(
            f"Expected 48 cache inputs and outputs, got {len(cache_inputs)} and {len(cache_outputs)}"
        )

    for value in cache_inputs:
        set_elem_type(value, TensorProto.FLOAT16)
    for value in cache_outputs:
        set_elem_type(value, TensorProto.FLOAT16)

    rewritten: list[onnx.NodeProto] = []
    gqa_count = 0
    for original in graph.node:
        node = copy.deepcopy(original)
        if node.domain == GQA_DOMAIN and node.op_type == GQA_OP:
            gqa_count += 1
            if not node.input or not node.input[0]:
                raise ValueError(f"{node.name} has no packed QKV input")
            if not node.output or not node.output[0]:
                raise ValueError(f"{node.name} has no attention output")

            suffix = clean(node.name or f"gqa_{gqa_count - 1}")
            qkv_fp32 = node.input[0]
            qkv_fp16 = f"{qkv_fp32}__native_fp16_{suffix}"
            attention_fp32 = node.output[0]
            attention_fp16 = f"{attention_fp32}__native_fp16_{suffix}"

            rewritten.append(
                helper.make_node(
                    "Cast",
                    [qkv_fp32],
                    [qkv_fp16],
                    name=f"native_gqa_qkv_to_fp16_{suffix}",
                    to=TensorProto.FLOAT16,
                )
            )
            node.input[0] = qkv_fp16
            node.output[0] = attention_fp16
            rewritten.append(node)
            rewritten.append(
                helper.make_node(
                    "Cast",
                    [attention_fp16],
                    [attention_fp32],
                    name=f"native_gqa_output_to_fp32_{suffix}",
                    to=TensorProto.FLOAT,
                )
            )
        else:
            rewritten.append(node)

    if gqa_count != 24:
        raise ValueError(f"Expected 24 GroupQueryAttention nodes, found {gqa_count}")

    graph.ClearField("node")
    graph.node.extend(rewritten)
    model.metadata_props.add(key="muscriptor.cache_storage", value="fp16_native_gqa")
    model.metadata_props.add(
        key="muscriptor.cache_strategy",
        value="fused GQA reads/writes FP16; only packed QKV and attention activation are cast",
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    base = onnx.load_model(args.decoder, load_external_data=False)
    output = args.output / "decoder-cache-fp16-native-gqa.onnx"
    model = convert(base)
    onnx.save_model(model, output, save_as_external_data=False)
    print(f"generated {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
