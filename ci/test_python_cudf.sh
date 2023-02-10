#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cudf"
pushd python/cudf/cudf
# (TODO: Copied the comment below from gpuCI, need to verify on GitHub Actions)
# It is essential to cd into python/cudf/cudf as `pytest-xdist` + `coverage` seem to work only at this directory level.
pytest \
  --cache-clear \
  --ignore="benchmarks" \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=../.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-coverage.xml" \
  --cov-report=term \
  tests
popd

# Run benchmarks with both cudf and pandas to ensure compatibility is maintained.
# Benchmarks are run in DEBUG_ONLY mode, meaning that only small data sizes are used.
# Therefore, these runs only verify that benchmarks are valid.
# They do not generate meaningful performance measurements.
pushd python/cudf
rapids-logger "pytest for cudf benchmarks"
CUDF_BENCHMARKS_DEBUG_ONLY=ON \
pytest \
  --cache-clear \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-coverage.xml" \
  --cov-report=term \
  benchmarks

rapids-logger "pytest for cudf benchmarks using pandas"
CUDF_BENCHMARKS_USE_PANDAS=ON \
CUDF_BENCHMARKS_DEBUG_ONLY=ON \
pytest \
  --cache-clear \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-pandas-coverage.xml" \
  --cov-report=term \
  benchmarks
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
