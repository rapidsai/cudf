#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist

# Download the cudf, libcudf, and pylibcudf built in the previous step
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

rapids-logger "Install dask_cudf, cudf, pylibcudf, and test requirements"

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_dask_cudf ./constraints.txt

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
  -v \
  --constraint ./constraints.txt \
  "$(echo ./dist/cudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
  "$(echo ./dist/dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
  "$(echo ./dist/libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
  "$(echo ./dist/pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl)"

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# Run tests in dask_cudf/tests and dask_cudf/io/tests
rapids-logger "pytest dask_cudf (dask-expr)"
pushd python/dask_cudf/dask_cudf
DASK_DATAFRAME__QUERY_PLANNING=True python -m pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  .
popd

# Run tests in dask_cudf/tests and dask_cudf/io/tests (legacy)
rapids-logger "pytest dask_cudf (legacy)"
pushd python/dask_cudf/dask_cudf
DASK_DATAFRAME__QUERY_PLANNING=False python -m pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-dask-cudf-legacy.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  .
popd
