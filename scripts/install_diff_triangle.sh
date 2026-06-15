#!/usr/bin/env bash
set -euo pipefail

submodule_dir="submodules/diff-triangle-rasterization"
patched_files=(
    "cuda_rasterizer/rasterizer_impl.h"
    "cuda_rasterizer/forward.h"
    "cuda_rasterizer/backward.h"
)

restore_patch() {
    if git -C "${submodule_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "${submodule_dir}" checkout -- "${patched_files[@]}" >/dev/null 2>&1 || true
    fi
}

trap restore_patch EXIT

bash scripts/patch_diff_triangle_cstdint.sh
python -m pip install --no-build-isolation "./${submodule_dir}"
