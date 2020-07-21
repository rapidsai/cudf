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
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Set Benchmark Vars
export ASVRESULTS_DIR=${WORKSPACE}/ci/artifacts/asv/results
export GBENCH_BENCHMARKS_DIR=${WORKSPACE}/cpp/build/gbenchmarks/

# Ensure ASV results directory exists
mkdir -p ${ASVRESULTS_DIR}

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
if [ "$PYTHON_VER" == "3.6" ];then
    conda install contextvars
fi

conda install "rmm=$MINOR_VERSION.*" "cudatoolkit=$CUDA_REL" \
              "rapids-build-env=$MINOR_VERSION.*" \
              "rapids-notebook-env=$MINOR_VERSION.*" \
              rapids-pytest-benchmark

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
# BUILD - Build libcudf, cuDF and dask_cudf from source
################################################################################

logger "Build libcudf..."
if [[ ${BUILD_MODE} == "pull-request" ]]; then
    $WORKSPACE/build.sh clean libcudf cudf dask_cudf benchmarks tests --ptds
else
    $WORKSPACE/build.sh clean libcudf cudf dask_cudf benchmarks tests -l --ptds
fi

################################################################################
# BENCHMARK - Run and parse libcudf and cuDF benchmarks
################################################################################

logger "Running benchmarks..."

#Download GBench results Parser
curl -L https://raw.githubusercontent.com/rapidsai/benchmark/main/parser/GBenchToASV.py --output GBenchToASV.py

mkdir -p ${WORKSPACE}/tmp/benchmark
touch ${WORKSPACE}/tmp/benchmark/benchmarks.txt
ls ${GBENCH_BENCHMARKS_DIR} > ${WORKSPACE}/tmp/benchmark/benchmarks.txt

#Disable error aborting while tests run, failed tests will not generate data
cd ${GBENCH_BENCHMARKS_DIR}
set +e
while read BENCH;
do
    nvidia-smi
    ./${BENCH} --benchmark_out=${BENCH}.json --benchmark_out_format=json
done < ${WORKSPACE}/tmp/benchmark/benchmarks.txt
set -e

rm ${WORKSPACE}/tmp/benchmark/benchmarks.txt
cd ${WORKSPACE}
mv ${GBENCH_BENCHMARKS_DIR}/*.json ${WORKSPACE}/tmp/benchmark/
python GBenchToASV.py -d ${WORKSPACE}/tmp/benchmark/ -t ${ASVRESULTS_DIR} -n libcudf -b branch-${MINOR_VERSION} 

