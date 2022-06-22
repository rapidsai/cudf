#!/bin/bash
set -e

# Update env vars
source rapids-env-update

# Check environment
source ci/check_environment.sh

# Use Ninja to build
export CMAKE_GENERATOR="Ninja"

################################################################################
# BUILD - Conda package builds (LIBCUGRAPH)
################################################################################
gpuci_logger "Begin cpp build"

CONDA_BLD_DIR="/tmp/conda-bld-workspace"

gpuci_mamba_retry mambabuild \
  --croot ${CONDA_BLD_DIR} \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
