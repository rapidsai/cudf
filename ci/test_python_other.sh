#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  dask-cudf cudf_kafka custreamz

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
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
popd

set -e
rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
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
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
