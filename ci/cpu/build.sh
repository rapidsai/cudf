#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
##############################################
# cuDF CPU conda build script for CI         #
##############################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package builds
################################################################################

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
  CONDA_BUILD_ARGS=""
  CONDA_CHANNEL=""
else
  CONDA_BUILD_ARGS="--dirty --no-remove-work-dir"
  CONDA_CHANNEL="-c $WORKSPACE/ci/artifacts/cudf/cpu/conda-bld/"
fi

if [ "$BUILD_LIBCUDF" == '1' ]; then
  gpuci_logger "Build conda pkg for libcudf"
  gpuci_conda_retry build conda/recipes/libcudf $CONDA_BUILD_ARGS

  gpuci_logger "Build conda pkg for libcudf_kafka"
  gpuci_conda_retry build conda/recipes/libcudf_kafka $CONDA_BUILD_ARGS
fi

if [ "$BUILD_CUDF" == '1' ]; then
  gpuci_logger "Build conda pkg for cudf"
  gpuci_conda_retry build conda/recipes/cudf --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL

  gpuci_logger "Build conda pkg for dask-cudf"
  gpuci_conda_retry build conda/recipes/dask-cudf --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL

  gpuci_logger "Build conda pkg for cudf_kafka"
  gpuci_conda_retry build conda/recipes/cudf_kafka --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL

  gpuci_logger "Build conda pkg for custreamz"
  gpuci_conda_retry build conda/recipes/custreamz --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL
fi
################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda pkgs"
source ci/cpu/upload.sh
