#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eou pipefail

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist

# Download libcudf and pylibcudf built in the previous step
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

rapids-logger "Installing cudf_polars and its dependencies"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cudf_polars ./constraints.txt

# echo to expand wildcard before adding `[test,experimental]` requires for pip
python -m pip install \
    -v \
    --constraint ./constraints.txt \
    "$(echo ./dist/cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test,experimental]" \
    "$(echo ./dist/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./dist/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)"

rapids-logger "Run cudf_polars tests"

function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

./ci/run_cudf_polars_pytests.sh \
       --cov cudf_polars \
       --cov-fail-under=100 \
       --cov-config=./pyproject.toml \
       --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars.xml"

trap ERR
set -e

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Testing FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Testing PASSED"
fi
exit ${EXITCODE}
