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

# Run the build via CMake, which will run clang-tidy when CUDF_STATIC_LINTERS is enabled.
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCUDF_STATIC_LINTERS=ON -GNinja
cmake --build cpp/build 2>&1 | python cpp/scripts/parse_iwyu_output.py

# Remove invalid components of the path for local usage. The path below is
# valid in the CI due to where the project is cloned, but presumably the fixes
# will be applied locally from inside a clone of cudf.
sed -i 's/\/__w\/cudf\/cudf\///' iwyu_results.txt
