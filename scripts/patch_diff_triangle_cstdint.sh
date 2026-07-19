#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

targets = [
    Path("submodules/diff-triangle-rasterization/cuda_rasterizer/rasterizer_impl.h"),
    Path("submodules/diff-triangle-rasterization/cuda_rasterizer/forward.h"),
    Path("submodules/diff-triangle-rasterization/cuda_rasterizer/backward.h"),
]

for path in targets:
    text = path.read_text()
    if "#include <cstdint>" in text:
        continue

    if "#include <vector>" in text:
        marker = "#include <vector>"
    elif "#include <cuda.h>" in text:
        marker = "#include <cuda.h>"
    else:
        raise SystemExit(f"Could not find include insertion point in {path}")

    path.write_text(text.replace(marker, f"{marker}\n #include <cstdint>", 1))
    print(f"patched {path}")
PY
