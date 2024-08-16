#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

# Download the pylibcudf built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-pylibcudf-dep
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Install both pylibcudf and cudf
python -m pip install \
    "$(echo ./local-pylibcudf-dep/pylibcudf*.whl)[test]" \
    "$(echo ./dist/cudf*.whl)[test]"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"


rapids-logger "pytest pylibcudf"
pushd python/pylibcudf/pylibcudf/tests
python -m pytest \
  --cache-clear \
  --dist=worksteal \
  .
popd

rapids-logger "pytest cudf"
pushd python/cudf/cudf/tests
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  .
popd
