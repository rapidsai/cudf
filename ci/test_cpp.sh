#!/bin/bash

set -euo pipefail

# TODO: Remove
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
source ci/check_env.sh

gpuci_logger "Check GPU usage"
nvidia-smi

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# Install libcudf packages
gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  libcudf libcudf_kafka libcudf-tests

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
TESTRESULTS_DIR=test-results
mkdir -p "${TESTRESULTS_DIR}"
SUITEERROR=0

set +e
for gt in "$CONDA_PREFIX/bin/gtests/libcudf"*/* ; do
  echo "Running GoogleTest ${gt}"
  ${gt} --gtest_output=xml:"${TESTRESULTS_DIR}/"
  EXITCODE=$?
    if (( ${EXITCODE} != 0 )); then
        SUITEERROR=${EXITECODE}
        echo "FAILED: GTest ${gt}"
    fi
done
set -e

if [ -n "${CODECOV_TOKEN}" ]; then
    codecov -t $CODECOV_TOKEN
fi

exit "${SUITEERROR}"
