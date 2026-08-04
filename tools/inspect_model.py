from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import onnx
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "happyme531/muscriptor-medium-onnx"
TEXT_SUFFIXES = {".md", ".json", ".py", ".txt", ".yaml", ".yml"}
MAX_TEXT_BYTES = 256_000
MAX_ONNX_BYTES = 1_500_000_000


def shape_of(value_info: onnx.ValueInfoProto) -> list[str | int]:
    tensor = value_info.type.tensor_type
    dims: list[str | int] = []
    for dim in tensor.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append("?")
    return dims


def describe_value(value_info: onnx.ValueInfoProto) -> dict[str, object]:
    tensor = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": shape_of(value_info),
    }


def main() -> None:
    api = HfApi()
    info = api.model_info(REPO_ID, files_metadata=True)
    print(f"repo={REPO_ID}")
    print(f"sha={info.sha}")
    print(f"private={info.private} gated={info.gated}")
    print("\nFILES")

    siblings = sorted(info.siblings or [], key=lambda item: item.rfilename)
    for sibling in siblings:
        size = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        print(json.dumps({"path": sibling.rfilename, "size": size, "lfs": lfs}, default=str))

    print("\nTEXT FILES")
    for sibling in siblings:
        suffix = Path(sibling.rfilename).suffix.lower()
        size = getattr(sibling, "size", None) or 0
        if suffix not in TEXT_SUFFIXES or size > MAX_TEXT_BYTES:
            continue
        try:
            local = Path(hf_hub_download(REPO_ID, sibling.rfilename))
            text = local.read_text(encoding="utf-8", errors="replace")
            print(f"\n--- {sibling.rfilename} ({len(text)} chars) ---")
            print(text)
        except Exception as exc:
            print(f"FAILED_TEXT {sibling.rfilename}: {exc!r}")

    print("\nONNX GRAPHS")
    for sibling in siblings:
        if not sibling.rfilename.lower().endswith(".onnx"):
            continue
        size = getattr(sibling, "size", None) or 0
        if size > MAX_ONNX_BYTES:
            print(f"SKIP_ONNX {sibling.rfilename}: {size} bytes")
            continue
        try:
            local = Path(hf_hub_download(REPO_ID, sibling.rfilename))
            model = onnx.load(str(local), load_external_data=False)
            op_counts = Counter(node.op_type for node in model.graph.node)
            external_initializers = []
            for init in model.graph.initializer:
                if init.data_location == onnx.TensorProto.EXTERNAL:
                    external_initializers.append(
                        {
                            "name": init.name,
                            "dims": list(init.dims),
                            "dtype": onnx.TensorProto.DataType.Name(init.data_type),
                            "external_data": {entry.key: entry.value for entry in init.external_data},
                        }
                    )
            report = {
                "file": sibling.rfilename,
                "ir_version": model.ir_version,
                "opsets": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
                "inputs": [describe_value(item) for item in model.graph.input],
                "outputs": [describe_value(item) for item in model.graph.output],
                "node_count": len(model.graph.node),
                "initializer_count": len(model.graph.initializer),
                "ops": dict(op_counts.most_common()),
                "external_initializer_count": len(external_initializers),
                "external_initializers_sample": external_initializers[:20],
            }
            print(json.dumps(report, indent=2))
        except Exception as exc:
            print(f"FAILED_ONNX {sibling.rfilename}: {exc!r}")


if __name__ == "__main__":
    main()
