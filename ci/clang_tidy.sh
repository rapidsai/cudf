#!/bin/bash
# Copyright (c) 2020-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n clang_tidy
conda activate clang_tidy

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

# Run the CMake configure step and set the build directory for clang-tidy.
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
export CUDF_ROOT="${PWD}/cpp/build"

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure --hook-stage manual
