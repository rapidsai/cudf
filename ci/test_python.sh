#!/bin/bash

set -euo pipefail

#TODO: Remove
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
source ci/check_env.sh

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  libcudf libcudf_kafka cudf dask-cudf cudf_kafka custreamz

# Install test dependencies
gpuci_mamba_retry install pytest pytest-cov pytest-xdist

gpuci_logger "Check GPU usage"
nvidia-smi

cd "${GITHUB_WORKSPACE}/python/cudf/cudf"
# It is essential to cd into ${GITHUB_WORKSPACE}/python/cudf/cudf as `pytest-xdist` + `coverage` seem to work only at this directory level.
gpuci_logger "Python py.test for cuDF"
py.test -n 8 --cache-clear --basetemp="${GITHUB_WORKSPACE}/cudf-cuda-tmp" --ignore="${GITHUB_WORKSPACE}/python/cudf/cudf/benchmarks" --junitxml="${GITHUB_WORKSPACE}/junit-cudf.xml" -v --cov-config="${GITHUB_WORKSPACE}/python/cudf/.coveragerc" --cov=cudf --cov-report=xml:"${GITHUB_WORKSPACE}/python/cudf/cudf-coverage.xml" --cov-report term --dist=loadscope tests

cd "${GITHUB_WORKSPACE}/python/dask_cudf"
gpuci_logger "Python py.test for dask-cudf"
py.test -n 8 --cache-clear --basetemp="${GITHUB_WORKSPACE}/dask-cudf-cuda-tmp" --junitxml="${GITHUB_WORKSPACE}/junit-dask-cudf.xml" -v --cov-config=.coveragerc --cov=dask_cudf --cov-report=xml:"${GITHUB_WORKSPACE}/python/dask_cudf/dask-cudf-coverage.xml" --cov-report term dask_cudf

cd "${GITHUB_WORKSPACE}/python/custreamz"
gpuci_logger "Python py.test for cuStreamz"
py.test -n 8 --cache-clear --basetemp="${GITHUB_WORKSPACE}/custreamz-cuda-tmp" --junitxml="${GITHUB_WORKSPACE}/junit-custreamz.xml" -v --cov-config=.coveragerc --cov=custreamz --cov-report=xml:"${GITHUB_WORKSPACE}/python/custreamz/custreamz-coverage.xml" --cov-report term custreamz

# Run benchmarks with both cudf and pandas to ensure compatibility is maintained.
# Benchmarks are run in DEBUG_ONLY mode, meaning that only small data sizes are used.
# Therefore, these runs only verify that benchmarks are valid.
# They do not generate meaningful performance measurements.
cd "${GITHUB_WORKSPACE}/python/cudf"
gpuci_logger "Python pytest for cuDF benchmarks"
CUDF_BENCHMARKS_DEBUG_ONLY=ON pytest -n 8 --cache-clear --basetemp="${GITHUB_WORKSPACE}/cudf-cuda-tmp" -v --dist=loadscope benchmarks

gpuci_logger "Python pytest for cuDF benchmarks using pandas"
CUDF_BENCHMARKS_USE_PANDAS=ON CUDF_BENCHMARKS_DEBUG_ONLY=ON pytest -n 8 --cache-clear --basetemp="${GITHUB_WORKSPACE}/cudf-cuda-tmp" -v --dist=loadscope benchmarks

gpuci_logger "Test notebooks"
"${GITHUB_WORKSPACE}/ci/test-notebooks.sh" 2>&1 | tee nbtest.log
python "${GITHUB_WORKSPACE}/ci/utils/nbtestlog2junitxml.py" nbtest.log

CODECOV_TOKEN="${CODECOV_TOKEN:-}"
if [ -n "${CODECOV_TOKEN}" ]; then
    codecov -t "${CODECOV_TOKEN}"
fi
