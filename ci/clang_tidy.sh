#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n clang_tidy

# Temporarily allow unbound variables for conda activation.
set +u
conda activate clang_tidy
set -u

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

source rapids-configure-sccache

# TODO: For testing purposes, clone and build IWYU. We can switch to a release
# once a clang 19-compatible version is available, which should be soon
# (https://github.com/include-what-you-use/include-what-you-use/issues/1641).
git clone https://github.com/include-what-you-use/include-what-you-use.git
pushd include-what-you-use
# IWYU's CMake build uses some Python scripts that assume that the cwd is
# importable, so support that legacy behavior.
export PYTHONPATH=${PWD}:${PYTHONPATH}
cmake -S . -B build -GNinja --install-prefix=${CONDA_PREFIX}
cmake --build build
cmake --install build
popd

# Run the build via CMake, which will run clang-tidy when CUDF_CLANG_TIDY is enabled.
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCUDF_CLANG_TIDY=ON -GNinja
cmake --build cpp/build
