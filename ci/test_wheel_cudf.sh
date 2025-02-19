#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the cudf, libcudf, and pylibcudf built in the previous step
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

rapids-logger "Install cudf, pylibcudf, and test requirements"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf ./constraints.txt

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
  "$(echo ./dist/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
  "$(echo ./dist/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo ./dist/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# Get the total GPU memory in MiB
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
GPU_MEMORY_GB=$((GPU_MEMORY / 1024))

# Set the NUM_PROCESSES based on GPU memory
if [ "$GPU_MEMORY_GB" -lt 24 ]; then
  NUM_PROCESSES=10
else
  NUM_PROCESSES=20
fi

rapids-logger "pytest pylibcudf"
pushd python/pylibcudf/pylibcudf/tests
python -m pytest \
  --cache-clear \
  --numprocesses=${NUM_PROCESSES} \
  --dist=worksteal \
  .
popd

rapids-logger "pytest cudf"
pushd python/cudf/cudf/tests
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
  --numprocesses=${NUM_PROCESSES} \
  --dist=worksteal \
  .
popd
