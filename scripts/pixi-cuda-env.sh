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

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export MAX_JOBS="${MAX_JOBS:-1}"
