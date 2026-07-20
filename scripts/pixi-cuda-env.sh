#!/usr/bin/env bash
set -euo pipefail

if [ -n "${CONDA_PREFIX:-}" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    cuda_target_dir="${CONDA_PREFIX}/targets/x86_64-linux"
    if [ -d "${cuda_target_dir}/include" ]; then
        export CPATH="${cuda_target_dir}/include${CPATH:+:${CPATH}}"
    fi
    if [ -d "${cuda_target_dir}/lib" ]; then
        export LD_LIBRARY_PATH="${cuda_target_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export LIBRARY_PATH="${cuda_target_dir}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    fi
fi

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    detected_compute_capability=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        detected_compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]' || true)"
    fi
    # 8.9 is the safe fallback for the RTX 4090 development host.  Systems
    # with a visible NVIDIA GPU use their queried capability instead, so the
    # same manifest also builds native Blackwell (12.0) extension binaries.
    export TORCH_CUDA_ARCH_LIST="${detected_compute_capability:-8.9}"
fi
export MAX_JOBS="${MAX_JOBS:-1}"
