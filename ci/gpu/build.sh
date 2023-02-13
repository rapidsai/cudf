#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

# Workaround to keep Jenkins builds working
# until we migrate fully to GitHub Actions
export RAPIDS_CUDA_VERSION="${CUDA}"
export SCCACHE_BUCKET=rapids-sccache
export SCCACHE_REGION=us-west-2
export SCCACHE_IDLE_TIMEOUT=32768

# Parse git describe
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
unset GIT_DESCRIBE_TAG

# Dask & Distributed option to install main(nightly) or `conda-forge` packages.
export INSTALL_DASK_MAIN=1

# Dask version to install when `INSTALL_DASK_MAIN=0`
export DASK_STABLE_VERSION="2022.12.0"

# ucx-py version
export UCX_PY_VERSION='0.31.*'

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

# Remove `dask/label/dev` channel if INSTALL_DASK_MAIN=0
if [ "$SOURCE_BRANCH" != "main" ] && [[ "${INSTALL_DASK_MAIN}" == 0 ]]; then
  conda config --system --remove channels dask/label/dev
  gpuci_mamba_retry install conda-forge::dask==$DASK_STABLE_VERSION conda-forge::distributed==$DASK_STABLE_VERSION conda-forge::dask-core==$DASK_STABLE_VERSION --force-reinstall
fi

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls
gpuci_logger "Check compiler versions"
python --version

function install_dask {
    # Install the conda-forge or nightly version of dask and distributed
    gpuci_logger "Install the conda-forge or nightly version of dask and distributed"
    set -x
    if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
        gpuci_logger "gpuci_mamba_retry install -c dask/label/dev 'dask/label/dev::dask' 'dask/label/dev::distributed'"
        gpuci_mamba_retry install -c dask/label/dev "dask/label/dev::dask" "dask/label/dev::distributed"
        conda list
    else
        gpuci_logger "gpuci_mamba_retry install conda-forge::dask=={$DASK_STABLE_VERSION} conda-forge::distributed=={$DASK_STABLE_VERSION} conda-forge::dask-core=={$DASK_STABLE_VERSION} --force-reinstall"
        gpuci_mamba_retry install conda-forge::dask==$DASK_STABLE_VERSION conda-forge::distributed==$DASK_STABLE_VERSION conda-forge::dask-core==$DASK_STABLE_VERSION --force-reinstall
    fi
    # Install the main version of streamz
    gpuci_logger "Install the main version of streamz"
    # Need to uninstall streamz that is already in the env.
    pip uninstall -y streamz
    pip install "git+https://github.com/python-streamz/streamz.git@master" --upgrade --no-deps
    set +x
}

install_dask

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then

    gpuci_logger "Install dependencies"
    gpuci_mamba_retry install -y \
                  "cudatoolkit=$CUDA_REL" \
                  "rapids-build-env=$MINOR_VERSION.*" \
                  "rapids-notebook-env=$MINOR_VERSION.*" \
                  "dask-cuda=${MINOR_VERSION}" \
                  "rmm=$MINOR_VERSION.*" \
                  "ucx-py=${UCX_PY_VERSION}"

    # https://docs.rapids.ai/maintainers/depmgmt/
    # gpuci_conda_retry remove --force rapids-build-env rapids-notebook-env
    # gpuci_mamba_retry install -y "your-pkg=1.0.0"

    ################################################################################
    # BUILD - Build libcudf, cuDF, libcudf_kafka, dask_cudf, and strings_udf from source
    ################################################################################

    gpuci_logger "Build from source"
    "$WORKSPACE/build.sh" clean libcudf cudf dask_cudf libcudf_kafka cudf_kafka strings_udf benchmarks tests --ptds

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
        cd "$WORKSPACE/cpp/build"

        for gt in "$WORKSPACE/cpp/build/gtests/"* ; do
            test_name=$(basename ${gt})
            echo "Running GoogleTest $test_name"
            ${gt} --gtest_output=xml:"$WORKSPACE/test-results/"
        done
    fi
else
    #Project Flash

    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
        exit 0
    fi

    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "Installing libcudf, libcudf_kafka and libcudf-tests"
    gpuci_mamba_retry install -y -c ${CONDA_ARTIFACT_PATH} libcudf libcudf_kafka libcudf-tests

    # TODO: Move boa install to gpuci/rapidsai
    gpuci_mamba_retry install boa
    gpuci_logger "Building cudf, dask-cudf, cudf_kafka and custreamz"
    export CONDA_BLD_DIR="$WORKSPACE/.conda-bld"
    gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/cudf --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}
    gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/dask-cudf --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}
    gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/cudf_kafka --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}
    gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/custreamz --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}

    # the CUDA component of strings_udf must be built on cuda 11.5 just like libcudf
    # but because there is no separate python package, we must also build the python on the 11.5 jobs
    # this means that at this point (on the GPU test jobs) the whole package is already built and has been
    # copied by CI from the upstream 11.5 jobs into $CONDA_ARTIFACT_PATH
    gpuci_logger "Installing cudf, dask-cudf, cudf_kafka, and custreamz"
    gpuci_mamba_retry install cudf dask-cudf cudf_kafka custreamz -c "${CONDA_BLD_DIR}" -c "${CONDA_ARTIFACT_PATH}"

    gpuci_logger "Check current conda environment"
    conda list --show-channel-urls

    gpuci_logger "GoogleTests"

    # Set up library for finding incorrect default stream usage.
    cd "$WORKSPACE/cpp/tests/utilities/identify_stream_usage/"
    mkdir build && cd build && cmake .. -GNinja && ninja && ninja test
    STREAM_IDENTIFY_LIB="$WORKSPACE/cpp/tests/utilities/identify_stream_usage/build/libidentify_stream_usage.so"

    # Run libcudf and libcudf_kafka gtests from libcudf-tests package
    for gt in "$CONDA_PREFIX/bin/gtests/libcudf"*/* ; do
        test_name=$(basename ${gt})

        echo "Running GoogleTest $test_name"
        if [[ ${test_name} == "SPAN_TEST" ]]; then
            # This one test is specifically designed to test using a thrust device
            # vector, so we expect and allow it to include default stream usage.
            gtest_filter="SpanTest.CanConstructFromDeviceContainers"
            GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:"$WORKSPACE/test-results/" --gtest_filter="-${gtest_filter}"
            ${gt} --gtest_output=xml:"$WORKSPACE/test-results/" --gtest_filter="${gtest_filter}"
        else
            GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:"$WORKSPACE/test-results/"
        fi
    done

    export LIB_BUILD_DIR="$WORKSPACE/ci/artifacts/cudf/cpu/libcudf_work/cpp/build"
    # Copy libcudf build time results
    echo "Checking for build time log $LIB_BUILD_DIR/ninja_log.xml"
    if [[ -f "$LIB_BUILD_DIR/ninja_log.xml" ]]; then
        gpuci_logger "Copying build time results"
        cp "$LIB_BUILD_DIR/ninja_log.xml" "$WORKSPACE/test-results/buildtimes-junit.xml"
    fi

    ################################################################################
    # MEMCHECK - Run compute-sanitizer on GoogleTest (only in nightly builds)
    ################################################################################
    if [[ "$BUILD_MODE" == "branch" && "$BUILD_TYPE" == "gpu" ]]; then
        if [[ "$COMPUTE_SANITIZER_ENABLE" == "true" ]]; then
            gpuci_logger "Memcheck on GoogleTests with rmm_mode=cuda"
            export GTEST_CUDF_RMM_MODE=cuda
            COMPUTE_SANITIZER_CMD="compute-sanitizer --tool memcheck"
            mkdir -p "$WORKSPACE/test-results/"
            for gt in "$CONDA_PREFIX/bin/gtests/libcudf"*/* ; do
                test_name=$(basename ${gt})
                if [[ "$test_name" == "ERROR_TEST" ]]; then
                  continue
                fi
                echo "Running GoogleTest $test_name"
                ${COMPUTE_SANITIZER_CMD} ${gt} | tee "$WORKSPACE/test-results/${test_name}.cs.log"
            done
            unset GTEST_CUDF_RMM_MODE
            # test-results/*.cs.log are processed in gpuci
        fi
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

cd "$WORKSPACE/python/cudf/cudf"
# It is essential to cd into $WORKSPACE/python/cudf/cudf as `pytest-xdist` + `coverage` seem to work only at this directory level.
gpuci_logger "Check conda packages"
conda list
gpuci_logger "Python py.test for cuDF"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/cudf-cuda-tmp" --ignore="$WORKSPACE/python/cudf/cudf/benchmarks" --junitxml="$WORKSPACE/junit-cudf.xml" -v --cov-config="$WORKSPACE/python/cudf/.coveragerc" --cov=cudf --cov-report=xml:"$WORKSPACE/python/cudf/cudf-coverage.xml" --cov-report term --dist=loadscope tests

gpuci_logger "Python py.tests for cuDF with spilling (CUDF_SPILL_DEVICE_LIMIT=1)"
# Due to time concerns, we only run tests marked "spilling"
CUDF_SPILL=on CUDF_SPILL_DEVICE_LIMIT=1 py.test -n 8 --cache-clear --basetemp="$WORKSPACE/cudf-cuda-tmp" --ignore="$WORKSPACE/python/cudf/cudf/benchmarks" -v --cov-config="$WORKSPACE/python/cudf/.coveragerc" --cov-append --cov=cudf --cov-report=xml:"$WORKSPACE/python/cudf/cudf-coverage.xml" --cov-report term --dist=loadscope -m spilling tests

cd "$WORKSPACE/python/dask_cudf"
gpuci_logger "Python py.test for dask-cudf"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/dask-cudf-cuda-tmp" --junitxml="$WORKSPACE/junit-dask-cudf.xml" -v --cov-config=.coveragerc --cov=dask_cudf --cov-report=xml:"$WORKSPACE/python/dask_cudf/dask-cudf-coverage.xml" --cov-report term dask_cudf

cd "$WORKSPACE/python/custreamz"
gpuci_logger "Python py.test for cuStreamz"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/custreamz-cuda-tmp" --junitxml="$WORKSPACE/junit-custreamz.xml" -v --cov-config=.coveragerc --cov=custreamz --cov-report=xml:"$WORKSPACE/python/custreamz/custreamz-coverage.xml" --cov-report term custreamz


# only install strings_udf after cuDF is finished testing without its presence
gpuci_logger "Installing strings_udf"
gpuci_mamba_retry install strings_udf -c "${CONDA_BLD_DIR}" -c "${CONDA_ARTIFACT_PATH}"

cd "$WORKSPACE/python/strings_udf/strings_udf"
gpuci_logger "Python py.test for strings_udf"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/strings-udf-cuda-tmp" --junitxml="$WORKSPACE/junit-strings-udf.xml" -v --cov-config=.coveragerc --cov=strings_udf --cov-report=xml:"$WORKSPACE/python/strings_udf/strings-udf-coverage.xml" --cov-report term tests

# retest cuDF UDFs
cd "$WORKSPACE/python/cudf/cudf"
gpuci_logger "Python py.test retest cuDF UDFs"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/cudf-cuda-strings-udf-tmp" --ignore="$WORKSPACE/python/cudf/cudf/benchmarks" --junitxml="$WORKSPACE/junit-cudf-strings-udf.xml" -v --cov-config="$WORKSPACE/python/cudf/.coveragerc" --cov=cudf --cov-report=xml:"$WORKSPACE/python/cudf/cudf-strings-udf-coverage.xml" --cov-report term --dist=loadscope tests/test_udf_masked_ops.py


# Run benchmarks with both cudf and pandas to ensure compatibility is maintained.
# Benchmarks are run in DEBUG_ONLY mode, meaning that only small data sizes are used.
# Therefore, these runs only verify that benchmarks are valid.
# They do not generate meaningful performance measurements.
cd "$WORKSPACE/python/cudf"
gpuci_logger "Python pytest for cuDF benchmarks"
CUDF_BENCHMARKS_DEBUG_ONLY=ON pytest -n 8 --cache-clear --basetemp="$WORKSPACE/cudf-cuda-tmp" -v --dist=loadscope benchmarks

gpuci_logger "Python pytest for cuDF benchmarks using pandas"
CUDF_BENCHMARKS_USE_PANDAS=ON CUDF_BENCHMARKS_DEBUG_ONLY=ON pytest -n 8 --cache-clear --basetemp="$WORKSPACE/cudf-cuda-tmp" -v --dist=loadscope benchmarks

gpuci_logger "Test notebooks"
"$WORKSPACE/ci/gpu/test-notebooks.sh" 2>&1 | tee nbtest.log
python "$WORKSPACE/ci/utils/nbtestlog2junitxml.py" nbtest.log

if [ -n "${CODECOV_TOKEN}" ]; then
    codecov -t $CODECOV_TOKEN
fi

return ${EXITCODE}
