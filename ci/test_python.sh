#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" > env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  cudf libcudf

TESTRESULTS_DIR="${PWD}/test-results"
mkdir -p "${TESTRESULTS_DIR}"
SUITEERROR=0

rapids-logger "Check GPU usage"
nvidia-smi

cd python

set +e

rapids-logger "pytest cudf"
# TODO: This needs copied from ci/gpu/build.sh. Paths are probably wrong.
pytest -n 8 --cache-clear --junitxml="${TESTRESULTS_DIR}/junit-cudf.xml" -v --cov-config=.coveragerc --cov=cudf --cov-report=xml:python/cudf-coverage.xml --cov-report term --dist=loadscope
exitcode=$?
if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in /cudf/python"
fi

exit ${SUITEERROR}
