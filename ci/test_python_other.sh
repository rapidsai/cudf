#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  dask-cudf cudf_kafka custreamz

TESTRESULTS_DIR="${PWD}/test-results"
mkdir -p "${TESTRESULTS_DIR}"
SUITEERROR=0

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "pytest dask_cudf"
pushd python/dask_cudf
pytest \
  --verbose \
  --cache-clear \
  --junitxml="${TESTRESULTS_DIR}/junit-dask-cudf.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=dask_cudf \
  --cov-report=xml:dask-cudf-coverage.xml \
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
  --verbose \
  --cache-clear \
  --junitxml="${TESTRESULTS_DIR}/junit-custreamz.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=custreamz \
  --cov-report=xml:custreamz-coverage.xml \
  --cov-report=term \
  custreamz
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in custreamz"
fi
popd

exit ${SUITEERROR}
