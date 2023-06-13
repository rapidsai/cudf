#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')
LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1223/8704a75/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
RMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1223/b8d1c12/rmm_conda_python_cuda${RAPIDS_CUDA_MAJOR}_3${PYTHON_MINOR_VERSION}_$(arch).tar.gz)
LIBKVIKIO_CHANNEL=$(rapids-get-artifact ci/kvikio/pull-request/224/68febbb/kvikio_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBKVIKIO_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/cudf

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBKVIKIO_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/dask-cudf

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBKVIKIO_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cudf_kafka

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBKVIKIO_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/custreamz


rapids-upload-conda-to-s3 python
