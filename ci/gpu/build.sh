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
# Set `LIBCUDF_KERNEL_CACHE_PATH` environment variable to $HOME/.jitify-cache because
# it's local to the container's virtual file system, and not shared with other CI jobs
# like `/tmp` is.
export LIBCUDF_KERNEL_CACHE_PATH="$HOME/.jitify-cache"

function remove_libcudf_kernel_cache_dir {
    EXITCODE=$?
    logger "removing kernel cache dir: $LIBCUDF_KERNEL_CACHE_PATH"
    rm -rf "$LIBCUDF_KERNEL_CACHE_PATH" || logger "could not rm -rf $LIBCUDF_KERNEL_CACHE_PATH"
    exit $EXITCODE
}

trap remove_libcudf_kernel_cache_dir EXIT

mkdir -p "$LIBCUDF_KERNEL_CACHE_PATH" || logger "could not mkdir -p $LIBCUDF_KERNEL_CACHE_PATH"

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf

# Install contextvars on Python 3.6
py_ver=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if [ "$py_ver" == "3.6" ];then
    conda install contextvars
fi

conda install "rmm=$MINOR_VERSION.*" "cudatoolkit=$CUDA_REL" \
              "rapids-build-env=$MINOR_VERSION.*" \
              "rapids-notebook-env=$MINOR_VERSION.*"

# https://docs.rapids.ai/maintainers/depmgmt/
# conda remove -f rapids-build-env rapids-notebook-env
# conda install "your-pkg=1.0.0"

# Install the master version of dask, distributed, and streamz
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps
logger "pip install git+https://github.com/python-streamz/streamz.git --upgrade --no-deps"
pip install "git+https://github.com/python-streamz/streamz.git" --upgrade --no-deps

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcudf, cuDF, libcudf_kafka, and dask_cudf from source
################################################################################

logger "Build libcudf..."
if [[ ${BUILD_MODE} == "pull-request" ]]; then
    $WORKSPACE/build.sh clean libcudf cudf dask_cudf libcudf_kafka cudf_kafka benchmarks tests --ptds
else
    $WORKSPACE/build.sh clean libcudf cudf dask_cudf libcudf_kafka cudf_kafka benchmarks tests -l --ptds
fi

################################################################################
# TEST - Run GoogleTest and py.tests for libcudf, and
# cuDF
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Check GPU usage..."
    nvidia-smi

    logger "GoogleTests..."
    cd $WORKSPACE/cpp/build

    for gt in ${WORKSPACE}/cpp/build/gtests/* ; do
        test_name=$(basename ${gt})
        echo "Running GoogleTest $test_name"
        ${gt} --gtest_output=xml:${WORKSPACE}/test-results/
    done

    # set environment variable for numpy 1.16
    # will be enabled for later versions by default
    np_ver=$(python -c "import numpy; print('.'.join(numpy.__version__.split('.')[:-1]))")
    if [ "$np_ver" == "1.16" ];then
        logger "export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1"
        export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
    fi

    cd $WORKSPACE/python/cudf
    logger "Python py.test for cuDF..."
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-cudf.xml -v --cov-config=.coveragerc --cov=cudf --cov-report=xml:${WORKSPACE}/python/cudf/cudf-coverage.xml --cov-report term

    cd $WORKSPACE/python/dask_cudf
    logger "Python py.test for dask-cudf..."
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-dask-cudf.xml -v --cov-config=.coveragerc --cov=dask_cudf --cov-report=xml:${WORKSPACE}/python/dask_cudf/dask-cudf-coverage.xml --cov-report term

    cd $WORKSPACE/python/custreamz
    logger "Python py.test for cuStreamz..."
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-custreamz.xml -v --cov-config=.coveragerc --cov=custreamz --cov-report=xml:${WORKSPACE}/python/custreamz/custreamz-coverage.xml --cov-report term

    ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
    python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log
fi
