#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../;

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_cudf

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest pylibcudf"
pushd python/pylibcudf/pylibcudf/tests
python -m pytest \
  --cache-clear \
  --dist=worksteal \
  .
popd

rapids-logger "pytest cudf"
./ci/run_cudf_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=../.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-coverage.xml" \
  --cov-report=term

# Run benchmarks with both cudf and pandas to ensure compatibility is maintained.
# Benchmarks are run in DEBUG_ONLY mode, meaning that only small data sizes are used.
# Therefore, these runs only verify that benchmarks are valid.
# They do not generate meaningful performance measurements.

rapids-logger "pytest for cudf benchmarks"
./ci/run_cudf_pytest_benchmarks.sh \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-coverage.xml" \
  --cov-report=term

rapids-logger "pytest for cudf benchmarks using pandas"
./ci/run_cudf_pandas_pytest_benchmarks.sh \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-pandas-coverage.xml" \
  --cov-report=term

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
