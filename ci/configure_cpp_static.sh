#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

source rapids-date-string

rapids-logger "Configure static cpp build"

ENV_YAML_DIR="$(mktemp -d)"
REQUIREMENTS_FILE="${ENV_YAML_DIR}/requirements.txt"

rapids-dependency-file-generator \
  --output requirements \
  --file_key test_static_build \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${REQUIREMENTS_FILE}"

python -m pip install -r "${REQUIREMENTS_FILE}"
pyenv rehash

cmake -S cpp -B build_static -GNinja -DBUILD_SHARED_LIBS=OFF -DCUDF_USE_ARROW_STATIC=ON -DBUILD_TESTS=OFF
