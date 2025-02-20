#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit;

# Common setup steps shared by Python test jobs
source ./ci/python/test/test_python_common.sh test_python_cudf

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Get the total GPU memory in MiB
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
GPU_MEMORY_GB=$((GPU_MEMORY / 1024))

# Set the NUM_PROCESSES based on GPU memory
if [ "$GPU_MEMORY_GB" -lt 24 ]; then
  NUM_PROCESSES=10
else
  NUM_PROCESSES=20
fi

rapids-logger "pytest pylibcudf"
./ci/python/test/run_pylibcudf_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibcudf.xml" \
  --numprocesses=${NUM_PROCESSES} \
  --dist=worksteal \
  --cov-config=../.coveragerc \
  --cov=pylibcudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibcudf-coverage.xml" \
  --cov-report=term

rapids-logger "pytest cudf"
./ci/python/test/run_cudf_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf.xml" \
  --numprocesses=${NUM_PROCESSES} \
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
./ci/python/test/run_cudf_pytest_benchmarks.sh \
  --numprocesses=${NUM_PROCESSES} \
  --dist=worksteal \
  --cov-config=.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-coverage.xml" \
  --cov-report=term

rapids-logger "pytest for cudf benchmarks using pandas"
./ci/python/test/run_cudf_pandas_pytest_benchmarks.sh \
  --numprocesses=${NUM_PROCESSES} \
  --dist=worksteal \
  --cov-config=.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-benchmark-pandas-coverage.xml" \
  --cov-report=term

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
