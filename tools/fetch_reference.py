from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "happyme531/muscriptor-medium-onnx"
FILES = [
    "README.md",
    "NOTICE.md",
    "runtime/LICENSE",
    "runtime/requirements.txt",
    "runtime/test_onnxruntime.py",
    "runtime/muscriptor_onnx/__init__.py",
    "runtime/muscriptor_onnx/audio.py",
    "onnx/w4a32_optimized/config.json",
]

root = Path("docs/model-reference")
for name in FILES:
    src = Path(hf_hub_download(REPO, name))
    dst = root / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print(f"copied {name} -> {dst}")
