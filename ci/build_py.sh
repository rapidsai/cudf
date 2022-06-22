#!/bin/bash
set -e

# Update env vars
source rapids-env-update

# Check environment
source ci/check_environment.sh

################################################################################
# BUILD - Conda package builds (CUGRAPH)
################################################################################
gpuci_logger "Begin py build"

# Python Build Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

gpuci_mamba_retry mambabuild \
  -c "${CPP_CHANNEL}" \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/cuml

gpuci_mamba_retry mambabuild \
  -c "${CPP_CHANNEL}" \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/dask-cudf

gpuci_mamba_retry mambabuild \
  -c "${CPP_CHANNEL}" \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/cudf_kafka

gpuci_mamba_retry mambabuild \
  -c "${CPP_CHANNEL}" \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/custreamz

rapids-upload-conda-to-s3 python
