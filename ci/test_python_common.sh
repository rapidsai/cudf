#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  cudf libcudf

COMMIT="99579d0"
LIBCUDF_CHANNEL_20=$(rapids-get-artifact ci/cudf/pull-request/13599/${COMMIT}/cudf_conda_cpp_cuda11_$(arch).tar.gz)
CUDF_CHANNEL_20=$(rapids-get-artifact ci/cudf/pull-request/13599/${COMMIT}/cudf_conda_python_cuda11_${RAPIDS_PY_VERSION//.}_$(arch).tar.gz)

rapids-logger $LIBCUDF_CHANNEL_20
rapids-logger $CUDF_CHANNEL_20

rapids-mamba-retry remove --force cudf libcudf dask-cudf pandas python-tzdata

rapids-mamba-retry install \
  --channel "${CUDF_CHANNEL_20}" \
  --channel "${LIBCUDF_CHANNEL_20}" \
  --channel dask/label/dev \
  --channel conda-forge \
  cudf libcudf dask-cudf pandas==2.0.2 python-tzdata
