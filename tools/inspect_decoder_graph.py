#!/usr/bin/env python3
"""Print the decoder attention/cache topology used by the native-cache experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load_model(args.decoder, load_external_data=False)
    print("opsets:", [(item.domain, item.version) for item in model.opset_import])
    counts = Counter((node.domain or "ai.onnx", node.op_type) for node in model.graph.node)
    print("operator counts:")
    for key, count in sorted(counts.items()):
        print(f"  {key[0]}::{key[1]} = {count}")

    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for value in node.input:
            if value:
                consumers[value].append(node)
        for value in node.output:
            if value:
                producers[value] = node

    print("cache paths:")
    for value in model.graph.input:
        if not value.name.startswith("past_"):
            continue
        uses = consumers.get(value.name, [])
        print(f"  {value.name}: {[f'{n.domain or "ai.onnx"}::{n.op_type}:{n.name}' for n in uses]}")

    print("attention-like nodes:")
    for index, node in enumerate(model.graph.node):
        label = f"{node.domain or 'ai.onnx'}::{node.op_type}"
        if any(word in node.op_type.lower() for word in ("attention", "flash", "rotary")):
            print(f"  [{index}] {label} name={node.name}")
            print(f"    inputs={list(node.input)}")
            print(f"    outputs={list(node.output)}")
            print(f"    attrs={[(a.name, onnx.helper.get_attribute_value(a)) for a in node.attribute]}")

    print("cache output producers:")
    for value in model.graph.output:
        if value.name.startswith("present_"):
            node = producers.get(value.name)
            print(f"  {value.name}: {None if node is None else f'{node.domain or "ai.onnx"}::{node.op_type}:{node.name}'}")


if __name__ == "__main__":
    main()
