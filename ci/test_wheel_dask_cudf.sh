#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# Download the cudf built in the previous step
# Set the manylinux version used for downloading the wheels so that we test the
# newer ABI wheels on the newer images that support their installation.
# Need to disable pipefail for the head not to fail, see
# https://stackoverflow.com/questions/19120263/why-exit-code-141-with-grep-q
set +o pipefail
glibc_minor_version=$(ldd --version | head -1 | grep -o "[0-9]\.[0-9]\+" | tail -1 | cut -d '.' -f2)
set -o pipefail
manylinux_version="2_17"
if [[ ${glibc_minor_version} -ge 28 ]]; then
    manylinux_version="2_28"
fi
manylinux="manylinux_${manylinux_version}"

RAPIDS_PY_WHEEL_NAME="cudf_${manylinux}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install --no-deps ./local-cudf-dep/cudf*.whl

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/dask_cudf*.whl)[test]

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
