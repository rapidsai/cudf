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
conda install "rmm=$MINOR_VERSION.*" "nvstrings=$MINOR_VERSION.*" "cudatoolkit=$CUDA_REL"

logger "Install Openjdk"
conda install -c anaconda openjdk

logger "Install maven"
conda install --no-deps -c conda-forge maven

logger "Check versions..."
python --version
$CC --version
$CXX --version
java -version
mvn -version
conda list

################################################################################
# BUILD - Build libcudf and cuDF from source
################################################################################

logger "Build libcudf..."
$WORKSPACE/build.sh clean libcudf cudf

################################################################################
# TEST - Run GoogleTest and py.tests for libcudf and cuDF
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
    exit 0
fi

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

logger "Python py.test for cuDF..."
cd $WORKSPACE/python
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cudf.xml -v --cov-config=.coveragerc --cov=cudf --cov-report=xml:${WORKSPACE}/cudf-coverage.xml --cov-report term

################################################################################
# TEST libcudfjni
################################################################################

logger "Test cudfjni"
cd $WORKSPACE/java
mvn -Dmaven.repo.local=$WORKSPACE/.m2 clean test
