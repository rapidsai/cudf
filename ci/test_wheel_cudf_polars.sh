#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eou pipefail

# We will only fail these tests if the PR touches code in pylibcudf
# or cudf_polars itself.
# Note, the three dots mean we are doing diff between the merge-base
# of upstream and HEAD. So this is asking, "does _this branch_ touch
# files in cudf_polars/pylibcudf", rather than "are there changes
# between upstream and this branch which touch cudf_polars/pylibcudf"
# TODO: is the target branch exposed anywhere in an environment variable?
if [ -n "$(git diff --name-only origin/branch-24.08...HEAD -- python/cudf_polars/ python/cudf/cudf/_lib/pylibcudf/)" ];
then
    HAS_CHANGES=1
else
    HAS_CHANGES=0
fi

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# Download the cudf built in the previous step
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install ./local-cudf-dep/cudf*.whl

rapids-logger "Install cudf_polars"
python -m pip install $(echo ./dist/cudf_polars*.whl)[test]

rapids-logger "Run cudf_polars tests"

function set_exitcode()
{
    EXITCODE=$?
}
EXITCODE=0
trap set_exitcode ERR
set +e

python -m pytest \
       --cache-clear \
       --cov cudf_polars \
       --cov-fail-under=100 \
       --cov-config=python/cudf_polars/pyproject.toml \
       --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars.xml" \
       python/cudf_polars/tests

trap ERR
set -e

if [ ${EXITCODE} != 0 ]; then
    rapids-logger "Testing FAILED: exitcode ${EXITCODE}"
else
    rapids-logger "Testing PASSED"
fi

if [ ${HAS_CHANGES} == 1 ]; then
    exit ${EXITCODE}
else
    exit 0
fi
