#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################################
# cuDF GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install "rmm=$MINOR_VERSION.*" "nvstrings=$MINOR_VERSION.*" "cudatoolkit=$CUDA_REL" \
              "dask>=2.0" "distributed>=2.0" "numpy>=1.16"

# Install the master version of dask and distributed
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps" 
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcudf and cuDF from source
################################################################################

logger "Build libcudf..."
$WORKSPACE/build.sh clean libcudf cudf dask_cudf

################################################################################
# TEST - Run GoogleTest and py.tests for libcudf and cuDF
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Check GPU usage..."
    nvidia-smi

    logger "GoogleTest for libcudf..."
    cd $WORKSPACE/cpp/build
    GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" make -j${PARALLEL_LEVEL} test

    # Install the master version of distributed for serialization testing
    logger "pip install git+https://github.com/dask/distributed.git"
    pip install "git+https://github.com/dask/distributed.git"

    # Temporarily install feather and cupy for testing
    logger "conda install feather-format"
    conda install "feather-format" "cupy>=6.0.0"

    # set environment variable for numpy 1.16
    # will be enabled for later versions by default
    np_ver=$(python -c "import numpy; print('.'.join(numpy.__version__.split('.')[:-1]))")
    if [ "$np_ver" == "1.16" ];then
      logger "export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1"
      export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
    fi


    logger "Python py.test for cuDF..."
    cd $WORKSPACE/python/cudf
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-cudf.xml -v --cov-config=.coveragerc --cov=cudf --cov-report=xml:${WORKSPACE}/python/cudf/cudf-coverage.xml --cov-report term
    
    cd $WORKSPACE/python/dask_cudf
    logger "Python py.test for dask-cudf..."
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-dask-cudf.xml -v --cov-config=.coveragerc --cov=dask_cudf --cov-report=xml:${WORKSPACE}/python/dask_cudf/dask-cudf-coverage.xml --cov-report term
fi
