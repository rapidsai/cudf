#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
##############################################
# cuDF GPU build and test script for CI      #
##############################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}
export CONDA_ARTIFACT_PATH="$WORKSPACE/ci/artifacts/cudf/cpu/.conda-bld/"

# Parse git describe
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# ucx-py version
export UCX_PY_VERSION='0.27.*'

################################################################################
# TRAP - Setup trap for removing jitify cache
################################################################################

# Set `LIBCUDF_KERNEL_CACHE_PATH` environment variable to $HOME/.jitify-cache
# because it's local to the container's virtual file system, and not shared with
# other CI jobs like `/tmp` is
export LIBCUDF_KERNEL_CACHE_PATH="$HOME/.jitify-cache"

function remove_libcudf_kernel_cache_dir {
    EXITCODE=$?
    gpuci_logger "TRAP: Removing kernel cache dir: $LIBCUDF_KERNEL_CACHE_PATH"
    rm -rf "$LIBCUDF_KERNEL_CACHE_PATH" \
        || gpuci_logger "[ERROR] TRAP: Could not rm -rf $LIBCUDF_KERNEL_CACHE_PATH"
    exit $EXITCODE
}

# Set trap to run on exit
gpuci_logger "TRAP: Set trap to remove jitify cache on exit"
trap remove_libcudf_kernel_cache_dir EXIT

mkdir -p "$LIBCUDF_KERNEL_CACHE_PATH" \
    || gpuci_logger "[ERROR] TRAP: Could not mkdir -p $LIBCUDF_KERNEL_CACHE_PATH"

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Install dependencies"
gpuci_mamba_retry install -y \
                  "cudatoolkit=$CUDA_REL" \
                  "rapids-build-env=$MINOR_VERSION.*" \
                  "rapids-notebook-env=$MINOR_VERSION.*" \
                  "dask-cuda=${MINOR_VERSION}" \
                  "rmm=$MINOR_VERSION.*" \
                  "ucx-py=${UCX_PY_VERSION}" \
                  "openjdk=8.*" \
                  "maven"
# "mamba install openjdk" adds an activation script to set JAVA_HOME but this is
# not triggered on installation. Re-activating the conda environment will set
# this environment variable so that CMake can find JNI.
conda activate rapids

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_conda_retry remove --force rapids-build-env rapids-notebook-env
# gpuci_mamba_retry install -y "your-pkg=1.0.0"


gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

function install_dask {
    # Install the main version of dask, distributed, and streamz
    gpuci_logger "Install the main version of dask, distributed, and streamz"
    set -x
    pip install "git+https://github.com/dask/distributed.git@main" --upgrade --no-deps
    pip install "git+https://github.com/dask/dask.git@main" --upgrade --no-deps
    # Need to uninstall streamz that is already in the env.
    pip uninstall -y streamz
    pip install "git+https://github.com/python-streamz/streamz.git@master" --upgrade --no-deps
    set +x
}

################################################################################
# INSTALL - Install libcudf artifacts
################################################################################

gpuci_logger "Installing libcudf & libcudf_kafka"
gpuci_mamba_retry install -c ${CONDA_ARTIFACT_PATH} libcudf libcudf_kafka

install_dask

################################################################################
# TEST - Run java tests
################################################################################

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Running Java Tests"
cd ${WORKSPACE}/java
mvn test -B -DCUDF_JNI_ARROW_STATIC=OFF

return ${EXITCODE}
