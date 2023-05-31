#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

PY_VER=${RAPIDS_PY_VERSION//./}
_CUDA_MAJOR=$(nvcc --version | tail -n 2 | head -n 1 | cut -d',' -f 2 | cut -d' ' -f 3 | cut -d'.' -f 1)

LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1278/4e1392d/rmm_conda_cpp_cuda${_CUDA_MAJOR}_$(arch).tar.gz)
RMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1278/4e1392d/rmm_conda_python_cuda${_CUDA_MAJOR}_${PY_VER}_$(arch).tar.gz)

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/cudf

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/dask-cudf

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/cudf_kafka

rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/custreamz


rapids-upload-conda-to-s3 python
