#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
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
export GBENCH_BENCHMARKS_DIR=${WORKSPACE}/cpp/build/gbenchmarks/

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
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Enter dependencies to be shown in ASV tooltips.
CUDF_DEPS=(librmm)
LIBCUDF_DEPS=(librmm)

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
conda info
conda config --show-sources
conda list --show-channel-urls

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

###
# Generate Metadata for dependencies
###

# Concatenate dependency arrays, convert to JSON array,
# and remove duplicates.
X=("${CUDF_DEPS[@]}" "${LIBCUDF_DEPS[@]}")
DEPS=$(printf '%s\n' "${X[@]}" | jq -R . | jq -s 'unique')

# Build object with k/v pairs of "dependency:version"
DEP_VER_DICT=$(jq -n '{}')
for DEP in $(echo "${DEPS}" | jq -r '.[]'); do
  VER=$(conda list | grep "^${DEP}" | awk '{print $2"-"$3}')
  DEP_VER_DICT=$(echo "${DEP_VER_DICT}" | jq -c --arg DEP "${DEP}" --arg VER "${VER}" '. + { ($DEP): $VER }')
done

# Pass in an array of dependencies to get a dict of "dependency:version"
function getReqs() {
  local DEPS_ARR=("$@")
  local REQS="{}"
  for DEP in "${DEPS_ARR[@]}"; do
    VER=$(echo "${DEP_VER_DICT}" | jq -r --arg DEP "${DEP}" '.[$DEP]')
    REQS=$(echo "${REQS}" | jq -c --arg DEP "${DEP}" --arg VER "${VER}" '. + { ($DEP): $VER }')
  done

  echo "${REQS}"
}

###
# Run LIBCUDF Benchmarks
###

REQS=$(getReqs "${LIBCUDF_DEPS[@]}")

mkdir -p ${WORKSPACE}/tmp/benchmark
touch ${WORKSPACE}/tmp/benchmark/benchmarks.txt
ls ${GBENCH_BENCHMARKS_DIR} > ${WORKSPACE}/tmp/benchmark/benchmarks.txt

#Disable error aborting while tests run, failed tests will not generate data
logger "Running libcudf GBenchmarks..."
cd ${GBENCH_BENCHMARKS_DIR}
set +e
while read BENCH;
do
    nvidia-smi
    ./${BENCH} --benchmark_out=${BENCH}.json --benchmark_out_format=json
    EXITCODE=$?
    if [[ ${EXITCODE} != 0 ]]; then
        rm ./${BENCH}.json
	JOBEXITCODE=1
    fi
done < ${WORKSPACE}/tmp/benchmark/benchmarks.txt
set -e

rm ${WORKSPACE}/tmp/benchmark/benchmarks.txt
cd ${WORKSPACE}
mv ${GBENCH_BENCHMARKS_DIR}/*.json ${WORKSPACE}/tmp/benchmark/
python GBenchToASV.py -d  ${WORKSPACE}/tmp/benchmark/ -t ${S3_ASV_DIR} -n libcudf -b branch-${MINOR_VERSION} -r "${REQS}"

###
# Run Python Benchmarks
###

#REQS=$(getReqs "${CUDF_DEPS[@]}")

#BENCHMARK_META=$(jq -n \
#  --arg NODE "${NODE_NAME}" \
#  --arg BRANCH "branch-${MINOR_VERSION}" \
#  --argjson REQS "${REQS}" '
#  {
#    "machineName": $NODE,
#    "commitBranch": $BRANCH,
#    "requirements": $REQS
#  }
#')

#echo "Benchmark meta:"
#echo "${BENCHMARK_META}" | jq "."
