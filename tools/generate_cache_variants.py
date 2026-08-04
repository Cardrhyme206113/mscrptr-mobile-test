#!/usr/bin/env python3
"""Generate small decoder graph variants with compressed persistent KV caches.

The base MuScriptor decoder consumes and produces FP32 past/present tensors. These adapters change
only the graph boundary representation. Each cache tensor is converted to FP32 immediately before
its layer uses it and converted back at the layer output, so model weights and arithmetic remain
unchanged while the long-lived Android direct buffers use FP16/BF16/INT8/packed INT4 storage.
"""

from __future__ import annotations

import argparse
import copy
import re
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto, helper

CACHE_INPUT = re.compile(r"^past_(?:key|value)\.\d+$")
CACHE_OUTPUT = re.compile(r"^present_(?:key|value)\.\d+$")


@dataclass(frozen=True)
class Variant:
    filename: str
    mode: str
    scale: float | None = None


VARIANTS = (
    Variant("decoder-cache-fp16.onnx", "fp16"),
    Variant("decoder-cache-bf16.onnx", "bf16"),
    Variant("decoder-cache-int8-balanced.onnx", "int8", 1.0 / 32.0),
    Variant("decoder-cache-int8-wide.onnx", "int8", 1.0 / 16.0),
    Variant("decoder-cache-int4-balanced.onnx", "int4", 0.5),
    Variant("decoder-cache-int4-wide.onnx", "int4", 1.0),
)


def clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def add_scalar(graph: onnx.GraphProto, name: str, data_type: int, value: int | float) -> str:
    graph.initializer.append(helper.make_tensor(name, data_type, [], [value]))
    return name


def add_vector(graph: onnx.GraphProto, name: str, values: list[int]) -> str:
    graph.initializer.append(helper.make_tensor(name, TensorProto.INT64, [len(values)], values))
    return name


def replace_node_inputs(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def rename_existing_value(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
        for index, value in enumerate(node.output):
            if value == old:
                node.output[index] = new
    for value_info in graph.value_info:
        if value_info.name == old:
            value_info.name = new


def set_value_info(value_info: onnx.ValueInfoProto, elem_type: int, packed: bool) -> None:
    tensor_type = value_info.type.tensor_type
    tensor_type.elem_type = elem_type
    if packed:
        dims = tensor_type.shape.dim
        if not dims:
            raise ValueError(f"Cache tensor {value_info.name} has no shape")
        last = dims[-1]
        last.ClearField("dim_param")
        last.dim_value = 32


def ensure_opset(model: onnx.ModelProto, minimum: int = 18) -> None:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            if opset.version < minimum:
                opset.version = minimum
            return
    model.opset_import.append(helper.make_opsetid("", minimum))


def add_float_input_adapter(graph: onnx.GraphProto, name: str, target_type: int) -> None:
    decoded = f"{name}__fp32"
    replace_node_inputs(graph, name, decoded)
    graph.node.append(
        helper.make_node(
            "Cast",
            [name],
            [decoded],
            name=f"cache_input_cast_{clean(name)}",
            to=TensorProto.FLOAT,
        )
    )
    value_info = next(value for value in graph.input if value.name == name)
    set_value_info(value_info, target_type, packed=False)


def add_float_output_adapter(graph: onnx.GraphProto, name: str, target_type: int) -> None:
    source = f"{name}__fp32"
    rename_existing_value(graph, name, source)
    graph.node.append(
        helper.make_node(
            "Cast",
            [source],
            [name],
            name=f"cache_output_cast_{clean(name)}",
            to=target_type,
        )
    )
    value_info = next(value for value in graph.output if value.name == name)
    set_value_info(value_info, target_type, packed=False)


def add_int8_input_adapter(graph: onnx.GraphProto, name: str, scale: float) -> None:
    prefix = clean(name)
    cast = f"{name}__int8_float"
    decoded = f"{name}__fp32"
    replace_node_inputs(graph, name, decoded)
    scale_name = add_scalar(graph, f"{prefix}_input_scale", TensorProto.FLOAT, scale)
    graph.node.extend(
        [
            helper.make_node(
                "Cast",
                [name],
                [cast],
                name=f"cache_input_int8_cast_{prefix}",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Mul",
                [cast, scale_name],
                [decoded],
                name=f"cache_input_int8_scale_{prefix}",
            ),
        ]
    )
    value_info = next(value for value in graph.input if value.name == name)
    set_value_info(value_info, TensorProto.INT8, packed=False)


def add_int8_output_adapter(graph: onnx.GraphProto, name: str, scale: float) -> None:
    prefix = clean(name)
    source = f"{name}__fp32"
    scaled = f"{name}__scaled"
    rounded = f"{name}__rounded"
    clipped = f"{name}__clipped"
    rename_existing_value(graph, name, source)
    scale_name = add_scalar(graph, f"{prefix}_output_scale", TensorProto.FLOAT, scale)
    min_name = add_scalar(graph, f"{prefix}_output_min", TensorProto.FLOAT, -128.0)
    max_name = add_scalar(graph, f"{prefix}_output_max", TensorProto.FLOAT, 127.0)
    graph.node.extend(
        [
            helper.make_node(
                "Div",
                [source, scale_name],
                [scaled],
                name=f"cache_output_int8_scale_{prefix}",
            ),
            helper.make_node(
                "Round",
                [scaled],
                [rounded],
                name=f"cache_output_int8_round_{prefix}",
            ),
            helper.make_node(
                "Clip",
                [rounded, min_name, max_name],
                [clipped],
                name=f"cache_output_int8_clip_{prefix}",
            ),
            helper.make_node(
                "Cast",
                [clipped],
                [name],
                name=f"cache_output_int8_cast_{prefix}",
                to=TensorProto.INT8,
            ),
        ]
    )
    value_info = next(value for value in graph.output if value.name == name)
    set_value_info(value_info, TensorProto.INT8, packed=False)


def add_int4_input_adapter(graph: onnx.GraphProto, name: str, scale: float) -> None:
    prefix = clean(name)
    low = f"{name}__low"
    shifted = f"{name}__high_shifted"
    high = f"{name}__high"
    low5 = f"{name}__low5"
    high5 = f"{name}__high5"
    paired = f"{name}__paired"
    shape = f"{name}__shape"
    prefix_shape = f"{name}__prefix_shape"
    unpack_shape = f"{name}__unpack_shape"
    unpacked = f"{name}__unpacked"
    cast = f"{name}__uint4_float"
    centered = f"{name}__centered"
    decoded = f"{name}__fp32"
    replace_node_inputs(graph, name, decoded)

    mask = add_scalar(graph, f"{prefix}_mask", TensorProto.UINT8, 15)
    shift = add_scalar(graph, f"{prefix}_shift", TensorProto.UINT8, 4)
    axes4 = add_vector(graph, f"{prefix}_axes4", [4])
    starts0 = add_vector(graph, f"{prefix}_shape_starts", [0])
    ends3 = add_vector(graph, f"{prefix}_shape_ends", [3])
    axes0 = add_vector(graph, f"{prefix}_shape_axes", [0])
    head_dim = add_vector(graph, f"{prefix}_head_dim", [64])
    zero_point = add_scalar(graph, f"{prefix}_zero_point", TensorProto.FLOAT, 8.0)
    scale_name = add_scalar(graph, f"{prefix}_input_scale", TensorProto.FLOAT, scale)

    graph.node.extend(
        [
            helper.make_node("BitwiseAnd", [name, mask], [low], name=f"cache_input_int4_low_{prefix}"),
            helper.make_node(
                "BitShift",
                [name, shift],
                [shifted],
                name=f"cache_input_int4_shift_{prefix}",
                direction="RIGHT",
            ),
            helper.make_node("BitwiseAnd", [shifted, mask], [high], name=f"cache_input_int4_high_{prefix}"),
            helper.make_node("Unsqueeze", [low, axes4], [low5], name=f"cache_input_int4_low_unsqueeze_{prefix}"),
            helper.make_node("Unsqueeze", [high, axes4], [high5], name=f"cache_input_int4_high_unsqueeze_{prefix}"),
            helper.make_node("Concat", [low5, high5], [paired], name=f"cache_input_int4_pair_{prefix}", axis=4),
            helper.make_node("Shape", [name], [shape], name=f"cache_input_int4_shape_{prefix}"),
            helper.make_node(
                "Slice",
                [shape, starts0, ends3, axes0],
                [prefix_shape],
                name=f"cache_input_int4_prefix_shape_{prefix}",
            ),
            helper.make_node(
                "Concat",
                [prefix_shape, head_dim],
                [unpack_shape],
                name=f"cache_input_int4_unpack_shape_{prefix}",
                axis=0,
            ),
            helper.make_node("Reshape", [paired, unpack_shape], [unpacked], name=f"cache_input_int4_reshape_{prefix}"),
            helper.make_node(
                "Cast",
                [unpacked],
                [cast],
                name=f"cache_input_int4_cast_{prefix}",
                to=TensorProto.FLOAT,
            ),
            helper.make_node("Sub", [cast, zero_point], [centered], name=f"cache_input_int4_center_{prefix}"),
            helper.make_node("Mul", [centered, scale_name], [decoded], name=f"cache_input_int4_scale_{prefix}"),
        ]
    )
    value_info = next(value for value in graph.input if value.name == name)
    set_value_info(value_info, TensorProto.UINT8, packed=True)


def add_int4_output_adapter(graph: onnx.GraphProto, name: str, scale: float) -> None:
    prefix = clean(name)
    source = f"{name}__fp32"
    scaled = f"{name}__scaled"
    rounded = f"{name}__rounded"
    clipped = f"{name}__clipped"
    biased = f"{name}__biased"
    unpacked = f"{name}__uint4"
    shape = f"{name}__shape"
    prefix_shape = f"{name}__prefix_shape"
    pack_shape = f"{name}__pack_shape"
    pairs = f"{name}__pairs"
    low5 = f"{name}__low5"
    high5 = f"{name}__high5"
    low = f"{name}__low"
    high = f"{name}__high"
    high_shifted = f"{name}__high_shifted"
    rename_existing_value(graph, name, source)

    scale_name = add_scalar(graph, f"{prefix}_output_scale", TensorProto.FLOAT, scale)
    min_name = add_scalar(graph, f"{prefix}_output_min", TensorProto.FLOAT, -8.0)
    max_name = add_scalar(graph, f"{prefix}_output_max", TensorProto.FLOAT, 7.0)
    zero_point = add_scalar(graph, f"{prefix}_output_zero_point", TensorProto.FLOAT, 8.0)
    starts0 = add_vector(graph, f"{prefix}_shape_starts", [0])
    ends3 = add_vector(graph, f"{prefix}_shape_ends", [3])
    axes0 = add_vector(graph, f"{prefix}_shape_axes", [0])
    packed_tail = add_vector(graph, f"{prefix}_packed_tail", [32, 2])
    slice_low_start = add_vector(graph, f"{prefix}_low_start", [0])
    slice_low_end = add_vector(graph, f"{prefix}_low_end", [1])
    slice_high_start = add_vector(graph, f"{prefix}_high_start", [1])
    slice_high_end = add_vector(graph, f"{prefix}_high_end", [2])
    axes4 = add_vector(graph, f"{prefix}_axes4", [4])
    shift = add_scalar(graph, f"{prefix}_shift", TensorProto.UINT8, 4)

    graph.node.extend(
        [
            helper.make_node("Div", [source, scale_name], [scaled], name=f"cache_output_int4_scale_{prefix}"),
            helper.make_node("Round", [scaled], [rounded], name=f"cache_output_int4_round_{prefix}"),
            helper.make_node(
                "Clip",
                [rounded, min_name, max_name],
                [clipped],
                name=f"cache_output_int4_clip_{prefix}",
            ),
            helper.make_node("Add", [clipped, zero_point], [biased], name=f"cache_output_int4_bias_{prefix}"),
            helper.make_node(
                "Cast",
                [biased],
                [unpacked],
                name=f"cache_output_int4_cast_{prefix}",
                to=TensorProto.UINT8,
            ),
            helper.make_node("Shape", [unpacked], [shape], name=f"cache_output_int4_shape_{prefix}"),
            helper.make_node(
                "Slice",
                [shape, starts0, ends3, axes0],
                [prefix_shape],
                name=f"cache_output_int4_prefix_shape_{prefix}",
            ),
            helper.make_node(
                "Concat",
                [prefix_shape, packed_tail],
                [pack_shape],
                name=f"cache_output_int4_pack_shape_{prefix}",
                axis=0,
            ),
            helper.make_node("Reshape", [unpacked, pack_shape], [pairs], name=f"cache_output_int4_reshape_{prefix}"),
            helper.make_node(
                "Slice",
                [pairs, slice_low_start, slice_low_end, axes4],
                [low5],
                name=f"cache_output_int4_low_slice_{prefix}",
            ),
            helper.make_node(
                "Slice",
                [pairs, slice_high_start, slice_high_end, axes4],
                [high5],
                name=f"cache_output_int4_high_slice_{prefix}",
            ),
            helper.make_node("Squeeze", [low5, axes4], [low], name=f"cache_output_int4_low_squeeze_{prefix}"),
            helper.make_node("Squeeze", [high5, axes4], [high], name=f"cache_output_int4_high_squeeze_{prefix}"),
            helper.make_node(
                "BitShift",
                [high, shift],
                [high_shifted],
                name=f"cache_output_int4_shift_{prefix}",
                direction="LEFT",
            ),
            helper.make_node("BitwiseOr", [low, high_shifted], [name], name=f"cache_output_int4_pack_{prefix}"),
        ]
    )
    value_info = next(value for value in graph.output if value.name == name)
    set_value_info(value_info, TensorProto.UINT8, packed=True)


def convert(base: onnx.ModelProto, variant: Variant) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    graph = model.graph
    ensure_opset(model)

    cache_inputs = [value.name for value in graph.input if CACHE_INPUT.match(value.name)]
    cache_outputs = [value.name for value in graph.output if CACHE_OUTPUT.match(value.name)]
    if len(cache_inputs) != 48 or len(cache_outputs) != 48:
        raise ValueError(
            f"Expected 48 cache inputs and outputs, got {len(cache_inputs)} and {len(cache_outputs)}"
        )

    for name in cache_inputs:
        if variant.mode == "fp16":
            add_float_input_adapter(graph, name, TensorProto.FLOAT16)
        elif variant.mode == "bf16":
            add_float_input_adapter(graph, name, TensorProto.BFLOAT16)
        elif variant.mode == "int8":
            assert variant.scale is not None
            add_int8_input_adapter(graph, name, variant.scale)
        elif variant.mode == "int4":
            assert variant.scale is not None
            add_int4_input_adapter(graph, name, variant.scale)
        else:
            raise ValueError(variant.mode)

    for name in cache_outputs:
        if variant.mode == "fp16":
            add_float_output_adapter(graph, name, TensorProto.FLOAT16)
        elif variant.mode == "bf16":
            add_float_output_adapter(graph, name, TensorProto.BFLOAT16)
        elif variant.mode == "int8":
            assert variant.scale is not None
            add_int8_output_adapter(graph, name, variant.scale)
        elif variant.mode == "int4":
            assert variant.scale is not None
            add_int4_output_adapter(graph, name, variant.scale)
        else:
            raise ValueError(variant.mode)

    model.metadata_props.add(key="muscriptor.cache_storage", value=variant.mode)
    if variant.scale is not None:
        model.metadata_props.add(key="muscriptor.cache_scale", value=str(variant.scale))
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    base = onnx.load_model(args.decoder, load_external_data=False)
    for variant in VARIANTS:
        output = args.output / variant.filename
        model = convert(base, variant)
        onnx.save_model(model, output, save_as_external_data=False)
        print(f"generated {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
