#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the cudf, libcudf, and pylibcudf built in the previous step
#CUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-logger "Install pylibcudf and its basic dependencies in a virtual environment"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf ./constraints.txt

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# To test pylibcudf without its optional dependencies, we create a virtual environment
python -m venv env
. env/bin/activate
rapids-pip-retry install \
    -v \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

rapids-pip-retry install pytest-repeat

rapids-logger "pytest pylibcudf without optional dependencies"
pushd python/pylibcudf/tests
python -m pytest \
  --cache-clear \
  --numprocesses=8 \
  --dist=worksteal \
  --count=300 \
  io/test_json.py::test_write_json_basic
popd

deactivate
