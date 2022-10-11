#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
##############################################
# cuDF CPU conda build script for CI         #
##############################################
set -e

# Set path and build parallel level
# FIXME: PATH variable shouldn't be necessary.
# This should be removed once we either stop using the `remote-docker-plugin`
# or the following issue is addressed: https://github.com/gpuopenanalytics/remote-docker-plugin/issues/47
export PATH=/usr/local/gcc9/bin:/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Use Ninja to build, setup Conda Build Dir
export CMAKE_GENERATOR="Ninja"
export CONDA_BLD_DIR="$WORKSPACE/.conda-bld"

# Whether to keep `dask/label/dev` channel in the env. If INSTALL_DASK_MAIN=0,
# `dask/label/dev` channel is removed.
export INSTALL_DASK_MAIN=1

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"

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

# Remove `rapidsai-nightly` & `dask/label/dev` channel if we are building main branch
if [ "$SOURCE_BRANCH" = "main" ]; then
  conda config --system --remove channels rapidsai-nightly
  conda config --system --remove channels dask/label/dev
elif [[ "${INSTALL_DASK_MAIN}" == 0 ]]; then
  # Remove `dask/label/dev` channel if INSTALL_DASK_MAIN=0
  conda config --system --remove channels dask/label/dev
fi

gpuci_logger "Check compiler versions"
python --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

# TODO: Move boa install to gpuci/rapidsai
gpuci_mamba_retry install boa

################################################################################
# BUILD - Conda package builds
################################################################################

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
  CONDA_BUILD_ARGS=""
  CONDA_CHANNEL=""
else
  CONDA_BUILD_ARGS="--dirty --no-remove-work-dir"
  CONDA_CHANNEL="-c $WORKSPACE/ci/artifacts/cudf/cpu/.conda-bld/"
fi

if [ "$BUILD_LIBCUDF" == '1' ]; then
  gpuci_logger "Build conda pkg for libcudf"
  gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcudf $CONDA_BUILD_ARGS

  # BUILD_LIBCUDF == 1 means this job is being run on the cpu_build jobs
  # that is where we must also build the strings_udf package
  mkdir -p ${CONDA_BLD_DIR}/strings_udf/work
  STRINGS_UDF_BUILD_DIR=${CONDA_BLD_DIR}/strings_udf/work
  gpuci_logger "Build conda pkg for cudf (python 3.8), for strings_udf"
  gpuci_conda_retry mambabuild --no-build-id --croot ${STRINGS_UDF_BUILD_DIR} -c ${CONDA_BLD_DIR} conda/recipes/cudf ${CONDA_BUILD_ARGS} --python=3.8
  gpuci_logger "Build conda pkg for cudf (python 3.9), for strings_udf"
  gpuci_conda_retry mambabuild --no-build-id --croot ${STRINGS_UDF_BUILD_DIR} -c ${CONDA_BLD_DIR} conda/recipes/cudf ${CONDA_BUILD_ARGS} --python=3.9

  gpuci_logger "Build conda pkg for strings_udf (python 3.8)"
  gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} -c ${STRINGS_UDF_BUILD_DIR} -c ${CONDA_BLD_DIR} conda/recipes/strings_udf $CONDA_BUILD_ARGS --python=3.8
  gpuci_logger "Build conda pkg for strings_udf (python 3.9)"
  gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} -c ${STRINGS_UDF_BUILD_DIR} -c ${CONDA_BLD_DIR} conda/recipes/strings_udf $CONDA_BUILD_ARGS --python=3.9

  mkdir -p ${CONDA_BLD_DIR}/libcudf/work
  cp -r ${CONDA_BLD_DIR}/work/* ${CONDA_BLD_DIR}/libcudf/work
  gpuci_logger "sccache stats"
  sccache --show-stats

  # Copy libcudf build metrics results
  LIBCUDF_BUILD_DIR=$CONDA_BLD_DIR/libcudf/work/cpp/build
  echo "Checking for build metrics log $LIBCUDF_BUILD_DIR/ninja_log.html"
  if [[ -f "$LIBCUDF_BUILD_DIR/ninja_log.html" ]]; then
      gpuci_logger "Copying build metrics results"
      mkdir -p "$WORKSPACE/build-metrics"
      cp "$LIBCUDF_BUILD_DIR/ninja_log.html" "$WORKSPACE/build-metrics/BuildMetrics.html"
      cp "$LIBCUDF_BUILD_DIR/ninja.log" "$WORKSPACE/build-metrics/ninja.log"
  fi
fi

if [ "$BUILD_CUDF" == '1' ]; then
  gpuci_logger "Build conda pkg for cudf"
  gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/cudf --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL

  gpuci_logger "Build conda pkg for dask-cudf"
  gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/dask-cudf --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL

  gpuci_logger "Build conda pkg for cudf_kafka"
  gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/cudf_kafka --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL

  gpuci_logger "Build conda pkg for custreamz"
  gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/custreamz --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL
  
  gpuci_logger "Build conda pkg for strings_udf"
  gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/strings_udf --python=$PYTHON $CONDA_BUILD_ARGS $CONDA_CHANNEL

fi
################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda pkgs"
source ci/cpu/upload.sh
