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
if [ -n "$(git diff --name-only origin/branch-24.10...HEAD -- python/cudf_polars/ python/pylibcudf/)" ];
then
    HAS_CHANGES=1
    rapids-logger "PR has changes in cudf-polars/pylibcudf, test fails treated as failure"
else
    HAS_CHANGES=0
    rapids-logger "PR does not have changes in cudf-polars/pylibcudf, test fails NOT treated as failure"
fi

rapids-logger "Download wheels"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist

# Download libcudf and pylibcudf built in the previous step
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

rapids-logger "Installing cudf_polars and its dependencies"
# Constraint to minimum dependency versions if job is set up as "oldest"
echo "" > ./constraints.txt
if [[ $RAPIDS_DEPENDENCIES == "oldest" ]]; then
    rapids-dependency-file-generator \
        --output requirements \
        --file-key py_test_cudf_polars \
        --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
      | tee ./constraints.txt
fi

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    -v \
    --constraint ./constraints.txt \
    "$(echo ./dist/cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
    "$(echo ./dist/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./dist/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)"

rapids-logger "Pin to 1.7.0 Temporarily"
python -m pip install polars==1.7.0

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

if [ ${HAS_CHANGES} == 1 ]; then
    exit ${EXITCODE}
else
    exit 0
fi
