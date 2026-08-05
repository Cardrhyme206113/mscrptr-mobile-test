#!/usr/bin/env python3
"""Rewrite MuScriptor's fused GQA nodes to the native K8/V8 custom operator.

The generated graph is incremental-only. The first 502-position audio-prefill is executed by the
existing native FP16 GQA graph; the resulting cache is quantized once. Subsequent one-token decoder
steps keep K and V as signed INT8 with one symmetric FP32 scale per head and position.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import onnx
from onnx import TensorProto, helper

CACHE_INPUT = re.compile(r"^past_(key|value)\.(\d+)$")
CACHE_OUTPUT = re.compile(r"^present_(key|value)\.(\d+)$")
GQA_DOMAIN = "com.microsoft"
GQA_OP = "GroupQueryAttention"
CUSTOM_DOMAIN = "dev.cardrhyme.kv"
CUSTOM_OP = "K8V8Attention"


def dims_of(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param") and dim.dim_param:
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def set_elem_type(value: onnx.ValueInfoProto, elem_type: int) -> None:
    value.type.tensor_type.elem_type = elem_type


def add_scale_value(
    collection: onnx.google.protobuf.internal.containers.RepeatedCompositeFieldContainer,
    name: str,
    cache_value: onnx.ValueInfoProto,
) -> None:
    shape = dims_of(cache_value)
    if len(shape) != 4:
        raise ValueError(f"Expected rank-4 cache tensor {cache_value.name}, got {shape}")
    collection.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape[:3]))


def convert(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    graph = model.graph

    cache_inputs = {value.name: value for value in graph.input if CACHE_INPUT.match(value.name)}
    cache_outputs = {value.name: value for value in graph.output if CACHE_OUTPUT.match(value.name)}
    if len(cache_inputs) != 48 or len(cache_outputs) != 48:
        raise ValueError(
            f"Expected 48 cache inputs and outputs, got {len(cache_inputs)} and {len(cache_outputs)}"
        )

    for value in cache_inputs.values():
        set_elem_type(value, TensorProto.INT8)
    for value in cache_outputs.values():
        set_elem_type(value, TensorProto.INT8)

    for layer in range(24):
        for kind in ("key", "value"):
            input_name = f"past_{kind}.{layer}"
            output_name = f"present_{kind}.{layer}"
            add_scale_value(graph.input, f"past_{kind}_scale.{layer}", cache_inputs[input_name])
            add_scale_value(graph.output, f"present_{kind}_scale.{layer}", cache_outputs[output_name])

    rewritten: list[onnx.NodeProto] = []
    count = 0
    for original in graph.node:
        if original.domain != GQA_DOMAIN or original.op_type != GQA_OP:
            rewritten.append(copy.deepcopy(original))
            continue

        if len(original.input) < 7 or len(original.output) < 3:
            raise ValueError(f"Unexpected GQA signature for {original.name}")
        match = re.match(r"^present_key\.(\d+)$", original.output[1])
        if not match:
            raise ValueError(f"Could not resolve layer from {original.output[1]}")
        layer = int(match.group(1))
        count += 1

        rewritten.append(
            helper.make_node(
                CUSTOM_OP,
                inputs=[
                    original.input[0],
                    f"past_key.{layer}",
                    f"past_key_scale.{layer}",
                    f"past_value.{layer}",
                    f"past_value_scale.{layer}",
                    original.input[5],
                    original.input[6],
                ],
                outputs=[
                    original.output[0],
                    f"present_key.{layer}",
                    f"present_key_scale.{layer}",
                    f"present_value.{layer}",
                    f"present_value_scale.{layer}",
                ],
                name=f"/layers.{layer}/K8V8Attention",
                domain=CUSTOM_DOMAIN,
                num_heads=16,
                head_size=64,
            )
        )

    if count != 24:
        raise ValueError(f"Expected 24 GroupQueryAttention nodes, found {count}")

    graph.ClearField("node")
    graph.node.extend(rewritten)
    if not any(item.domain == CUSTOM_DOMAIN for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid(CUSTOM_DOMAIN, 1))
    model.metadata_props.add(key="muscriptor.cache_storage", value="k8v8_native")
    model.metadata_props.add(
        key="muscriptor.cache_strategy",
        value="FP16 prefill once; incremental fused INT8 K/V attention with per-token/head scales",
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    base = onnx.load_model(args.decoder, load_external_data=False)
    output = args.output / "decoder-cache-k8v8-native.onnx"
    onnx.save_model(convert(base), output, save_as_external_data=False)
    print(f"generated {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
