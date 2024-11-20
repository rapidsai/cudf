#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eou pipefail

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# Download libcudf and pylibcudf built in the previous step
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libcudf-dep
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./local-pylibcudf-dep

rapids-logger "Install libcudf, pylibcudf and cudf_polars"
python -m pip install \
    -v \
    "$(echo ./dist/cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
    "$(echo ./local-libcudf-dep/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./local-pylibcudf-dep/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)"


TAG=$(python -c 'import polars; print(f"py-{polars.__version__}")')
rapids-logger "Clone polars to ${TAG}"
git clone https://github.com/pola-rs/polars.git --branch ${TAG} --depth 1

# Install requirements for running polars tests
rapids-logger "Install polars test requirements"
python -m pip install -r polars/py-polars/requirements-dev.txt -r polars/py-polars/requirements-ci.txt

function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

rapids-logger "Run polars tests"
./ci/run_cudf_polars_polars_tests.sh

trap ERR
set -e

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Running polars test suite FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Running polars test suite PASSED"
fi
exit ${EXITCODE}
