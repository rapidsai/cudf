#!/bin/bash

set -euo pipefail

# TODO: Move in recipe build?
export CMAKE_GENERATOR=Ninja

# TODO: Move to job config
export CUDA=11.5

# Update env vars
source rapids-env-update

# Check environment
source ci/check_env.sh

################################################################################
# BUILD - Conda package builds (CUDF)
################################################################################

gpuci_logger "Begin py build"

# Python Build Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

gpuci_mamba_retry mambabuild -c "${CPP_CHANNEL}" conda/recipes/cudf

rapids-upload-conda-to-s3 python
