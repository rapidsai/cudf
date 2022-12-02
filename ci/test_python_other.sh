#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  dask-cudf cudf_kafka custreamz

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "pytest dask_cudf"
pushd python/dask_cudf
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=dask_cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/dask-cudf-coverage.xml" \
  --cov-report=term \
  dask_cudf
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in dask-cudf"
fi
popd

rapids-logger "pytest custreamz"
pushd python/custreamz
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-custreamz.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=custreamz \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/custreamz-coverage.xml" \
  --cov-report=term \
  custreamz
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in custreamz"
fi
popd

set -e
rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  strings_udf
set +e

rapids-logger "pytest strings_udf"
pushd python/strings_udf/strings_udf
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-strings-udf.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=strings_udf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/strings-udf-coverage.xml" \
  --cov-report=term \
  tests
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in strings_udf"
fi
popd

rapids-logger "pytest cudf with strings_udf"
pushd python/cudf/cudf
pytest \
  --cache-clear \
  --ignore="benchmarks" \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-strings-udf.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=../.coveragerc \
  --cov=cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-strings-udf-coverage.xml" \
  --cov-report=term \
  tests/test_udf_masked_ops.py
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in cudf with strings_udf"
fi
popd

exit ${SUITEERROR}
