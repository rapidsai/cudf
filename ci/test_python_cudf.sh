#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  cudf libcudf

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "pytest cudf"
pushd python/cudf/cudf
# (TODO: Copied the comment below from gpuCI, need to verify on GitHub Actions)
# It is essential to cd into python/cudf/cudf as `pytest-xdist` + `coverage` seem to work only at this directory level.
pytest \
  --verbose \
  --cache-clear \
  --ignore="benchmarks" \
  --junitxml="${TESTRESULTS_DIR}/junit-cudf.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=../.coveragerc \
  --cov=cudf \
  --cov-report=xml:cudf-coverage.xml \
  --cov-report=term \
  tests
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in cudf"
fi
popd

exit ${SUITEERROR}
