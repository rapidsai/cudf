#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# Download the cudf built in the previous step
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    "$(echo ./dist/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./dist/cudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./dist/dask_cudf*.whl)[test]"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# Run tests in dask_cudf/tests and dask_cudf/io/tests
rapids-logger "pytest dask_cudf (dask-expr)"
pushd python/dask_cudf/dask_cudf
DASK_DATAFRAME__QUERY_PLANNING=True python -m pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf.xml" \
  --numprocesses=8 \
  .
popd

# Run tests in dask_cudf/tests and dask_cudf/io/tests (legacy)
rapids-logger "pytest dask_cudf (legacy)"
pushd python/dask_cudf/dask_cudf
DASK_DATAFRAME__QUERY_PLANNING=False python -m pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf-legacy.xml" \
  --numprocesses=8 \
  .
popd
