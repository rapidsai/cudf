#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Parse git describe
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

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
gpuci_conda_retry install -y \
                  "cudatoolkit=$CUDA_REL" \
                  "rapids-build-env=$MINOR_VERSION.*" \
                  "rapids-notebook-env=$MINOR_VERSION.*" \
                  "dask-cuda=${MINOR_VERSION}" \
                  "rmm=$MINOR_VERSION.*" \
                  "ucx-py=${MINOR_VERSION}"

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_conda_retry remove --force rapids-build-env rapids-notebook-env
gpuci_conda_retry install -y "numpy<1.20"


gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

function install_dask {
    # Install the master version of dask, distributed, and streamz
    gpuci_logger "Install the master version of dask, distributed, and streamz"
    set -x
    pip install "git+https://github.com/dask/distributed.git@master" --upgrade --no-deps
    pip install "git+https://github.com/dask/dask.git@master" --upgrade --no-deps
    pip install "git+https://github.com/python-streamz/streamz.git" --upgrade --no-deps
    set +x
}

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then

    install_dask

    ################################################################################
    # BUILD - Build libcudf, cuDF, libcudf_kafka, and dask_cudf from source
    ################################################################################

    gpuci_logger "Build from source"
    if [[ ${BUILD_MODE} == "pull-request" ]]; then
        $WORKSPACE/build.sh clean libcudf cudf dask_cudf libcudf_kafka cudf_kafka benchmarks tests --ptds
    else
        $WORKSPACE/build.sh clean libcudf cudf dask_cudf libcudf_kafka cudf_kafka benchmarks tests -l --ptds
    fi

    ################################################################################
    # TEST - Run GoogleTest
    ################################################################################

    set +e -Eo pipefail
    EXITCODE=0
    trap "EXITCODE=1" ERR


    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
        exit 0
    else
        gpuci_logger "Check GPU usage"
        nvidia-smi

        gpuci_logger "GoogleTests"
        set -x
        cd $WORKSPACE/cpp/build

        for gt in ${WORKSPACE}/cpp/build/gtests/* ; do
            test_name=$(basename ${gt})
            echo "Running GoogleTest $test_name"
            ${gt} --gtest_output=xml:${WORKSPACE}/test-results/
        done
    fi
else
    #Project Flash
    export LIB_BUILD_DIR="$WORKSPACE/ci/artifacts/cudf/cpu/libcudf_work/cpp/build"
    export LD_LIBRARY_PATH="$LIB_BUILD_DIR:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    
    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
        exit 0
    fi

    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "GoogleTests"
    set -x
    cd $LIB_BUILD_DIR

    for gt in gtests/* ; do
        test_name=$(basename ${gt})
        echo "Running GoogleTest $test_name"
        ${gt} --gtest_output=xml:${WORKSPACE}/test-results/
    done

    CUDF_CONDA_FILE=`find $WORKSPACE/ci/artifacts/cudf/cpu/conda-bld/ -name "libcudf-*.tar.bz2"`
    CUDF_CONDA_FILE=`basename "$CUDF_CONDA_FILE" .tar.bz2` #get filename without extension
    CUDF_CONDA_FILE=${CUDF_CONDA_FILE//-/=} #convert to conda install
    KAFKA_CONDA_FILE=`find $WORKSPACE/ci/artifacts/cudf/cpu/conda-bld/ -name "libcudf_kafka-*.tar.bz2"`
    KAFKA_CONDA_FILE=`basename "$KAFKA_CONDA_FILE" .tar.bz2` #get filename without extension
    KAFKA_CONDA_FILE=${KAFKA_CONDA_FILE//-/=} #convert to conda install

    gpuci_logger "Installing $CUDF_CONDA_FILE & $KAFKA_CONDA_FILE"
    conda install -c $WORKSPACE/ci/artifacts/cudf/cpu/conda-bld/ "$CUDF_CONDA_FILE" "$KAFKA_CONDA_FILE"

    install_dask

    gpuci_logger "Build python libs from source"
    if [[ ${BUILD_MODE} == "pull-request" ]]; then
        $WORKSPACE/build.sh cudf dask_cudf cudf_kafka --ptds
    else
        $WORKSPACE/build.sh cudf dask_cudf cudf_kafka -l --ptds
    fi
fi

# Both regular and Project Flash proceed here

# set environment variable for numpy 1.16
# will be enabled for later versions by default
np_ver=$(python -c "import numpy; print('.'.join(numpy.__version__.split('.')[:-1]))")
if [ "$np_ver" == "1.16" ];then
    export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
fi


################################################################################
# TEST - Run py.test, notebooks
################################################################################

cd $WORKSPACE/python/cudf
gpuci_logger "Python py.test for cuDF"
py.test -n 6 --cache-clear --basetemp=${WORKSPACE}/cudf-cuda-tmp --junitxml=${WORKSPACE}/junit-cudf.xml -v --cov-config=.coveragerc --cov=cudf --cov-report=xml:${WORKSPACE}/python/cudf/cudf-coverage.xml --cov-report term

cd $WORKSPACE/python/dask_cudf
gpuci_logger "Python py.test for dask-cudf"
py.test -n 6 --cache-clear --basetemp=${WORKSPACE}/dask-cudf-cuda-tmp --junitxml=${WORKSPACE}/junit-dask-cudf.xml -v --cov-config=.coveragerc --cov=dask_cudf --cov-report=xml:${WORKSPACE}/python/dask_cudf/dask-cudf-coverage.xml --cov-report term

cd $WORKSPACE/python/custreamz
gpuci_logger "Python py.test for cuStreamz"
py.test -n 6 --cache-clear --basetemp=${WORKSPACE}/custreamz-cuda-tmp --junitxml=${WORKSPACE}/junit-custreamz.xml -v --cov-config=.coveragerc --cov=custreamz --cov-report=xml:${WORKSPACE}/python/custreamz/custreamz-coverage.xml --cov-report term

gpuci_logger "Test notebooks"
${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log

return ${EXITCODE}