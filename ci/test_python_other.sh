#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_other

RAPIDS_VERSION="$(rapids-version)"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  "dask-cudf=${RAPIDS_VERSION}" \
  "cudf_kafka=${RAPIDS_VERSION}" \
  "custreamz=${RAPIDS_VERSION}" \
  "cudf-polars=${RAPIDS_VERSION}"

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest dask_cudf (dask-expr)"
DASK_DATAFRAME__QUERY_PLANNING=True ./ci/run_dask_cudf_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=../.coveragerc \
  --cov=dask_cudf \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/dask-cudf-coverage.xml" \
  --cov-report=term

rapids-logger "pytest dask_cudf (legacy)"
DASK_DATAFRAME__QUERY_PLANNING=False ./ci/run_dask_cudf_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf-legacy.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  .

rapids-logger "pytest cudf_kafka"
./ci/run_cudf_kafka_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-kafka.xml"

rapids-logger "pytest custreamz"
./ci/run_custreamz_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-custreamz.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=../.coveragerc \
  --cov=custreamz \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/custreamz-coverage.xml" \
  --cov-report=term

# Note that cudf-polars uses rmm.mr.CudaAsyncMemoryResource() which allocates
# half the available memory. This doesn't play well with multiple workers, so
# we keep --numprocesses=1 for now. This should be resolved by
# https://github.com/rapidsai/cudf/issues/16723.
rapids-logger "pytest cudf-polars"
./ci/run_cudf_polars_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars.xml" \
  --numprocesses=1 \
  --dist=worksteal \
  --cov-config=./pyproject.toml \
  --cov=cudf_polars \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-polars-coverage.xml" \
  --cov-report=term

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
