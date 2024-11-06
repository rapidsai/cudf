#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the cudf, libcudf, and pylibcudf built in the previous step
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

rapids-logger "Install cudf, pylibcudf, and test requirements"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf ./constraints.txt

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    -v \
    --constraint ./constraints.txt \
  "$(echo ./dist/cudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
  "$(echo ./dist/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
  "$(echo ./dist/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"


rapids-logger "pytest pylibcudf"
pushd python/pylibcudf/pylibcudf/tests
python -m pytest \
  --cache-clear \
  --numprocesses=8 \
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
