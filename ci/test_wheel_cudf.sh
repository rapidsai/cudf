#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the cudf, libcudf, and pylibcudf built in the previous step
CUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Install cudf, pylibcudf, and test requirements"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf ./constraints.txt

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUDF_WHEELHOUSE}"/cudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

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
