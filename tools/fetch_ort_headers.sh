#!/usr/bin/env bash
set -euo pipefail

root="${1:-app/src/main/cpp/ort_headers}"
version="v1.26.0"
base="https://raw.githubusercontent.com/microsoft/onnxruntime/${version}/include/onnxruntime/core/session"
mkdir -p "$root"

for header in \
  onnxruntime_c_api.h \
  onnxruntime_ep_c_api.h \
  onnxruntime_cxx_api.h \
  onnxruntime_cxx_inline.h \
  onnxruntime_float16.h
do
  target="$root/$header"
  if [[ ! -s "$target" ]]; then
    curl -L --fail --retry 4 --retry-delay 2 -o "$target.tmp" "$base/$header"
    mv "$target.tmp" "$target"
  fi
done
