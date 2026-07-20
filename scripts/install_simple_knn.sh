#!/usr/bin/env bash
set -euo pipefail

submodule_dir="submodules/simple-knn"

cleanup_build_artifacts() {
    rm -rf -- "${submodule_dir}/build" "${submodule_dir}/simple_knn.egg-info"
}

# setuptools leaves both directories in the submodule. They are not needed
# after the wheel is installed into Pixi's environment and should not make the
# parent checkout look dirty.
trap cleanup_build_artifacts EXIT
cleanup_build_artifacts

python -m pip install --force-reinstall --no-build-isolation "./${submodule_dir}"
