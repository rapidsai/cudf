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
unset GIT_DESCRIBE_TAG

# Dask & Distributed option to install main(nightly) or `conda-forge` packages.
export INSTALL_DASK_MAIN=1

# ucx-py version
export UCX_PY_VERSION='0.26.*'

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

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

function install_dask {
    # Install the conda-forge or nightly version of dask and distributed
    gpuci_logger "Install the conda-forge or nightly version of dask and distributed"
    set -x
    if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
        gpuci_logger "gpuci_mamba_retry update dask"
        gpuci_mamba_retry update dask
        conda list
    else
        gpuci_logger "gpuci_mamba_retry install conda-forge::dask>=2022.03.0 conda-forge::distributed>=2022.03.0 conda-forge::dask-core>=2022.03.0 --force-reinstall"
        gpuci_mamba_retry install conda-forge::dask>=2022.03.0 conda-forge::distributed>=2022.03.0 conda-forge::dask-core>=2022.03.0 --force-reinstall
    fi
    # Install the main version of streamz
    gpuci_logger "Install the main version of streamz"
    # Need to uninstall streamz that is already in the env.
    pip uninstall -y streamz
    pip install "git+https://github.com/python-streamz/streamz.git@master" --upgrade --no-deps
    set +x
}

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

    install_dask

    ################################################################################
    # BUILD - Build libcudf, cuDF, libcudf_kafka, and dask_cudf from source
    ################################################################################

    gpuci_logger "Build from source"
    "$WORKSPACE/build.sh" clean libcudf cudf dask_cudf libcudf_kafka cudf_kafka benchmarks tests --ptds

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

    gpuci_logger "Building cudf, dask-cudf, cudf_kafka and custreamz"
    export CONDA_BLD_DIR="$WORKSPACE/.conda-bld"
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/cudf --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/dask-cudf --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/cudf_kafka --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/custreamz --python=$PYTHON -c ${CONDA_ARTIFACT_PATH}

    gpuci_logger "Installing cudf, dask-cudf, cudf_kafka and custreamz"
    gpuci_mamba_retry install cudf dask-cudf cudf_kafka custreamz -c "${CONDA_BLD_DIR}" -c "${CONDA_ARTIFACT_PATH}"

    gpuci_logger "GoogleTests"
    # Run libcudf and libcudf_kafka gtests from libcudf-tests package
    for gt in "$CONDA_PREFIX/bin/gtests/libcudf"*/* ; do
        echo "Running GoogleTest $test_name"
        ${gt} --gtest_output=xml:"$WORKSPACE/test-results/"
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
gpuci_logger "Python py.test for cuDF"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/cudf-cuda-tmp" --ignore="$WORKSPACE/python/cudf/cudf/benchmarks" --junitxml="$WORKSPACE/junit-cudf.xml" -v --cov-config="$WORKSPACE/python/cudf/.coveragerc" --cov=cudf --cov-report=xml:"$WORKSPACE/python/cudf/cudf-coverage.xml" --cov-report term --dist=loadscope tests

cd "$WORKSPACE/python/dask_cudf"
gpuci_logger "Python py.test for dask-cudf"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/dask-cudf-cuda-tmp" --junitxml="$WORKSPACE/junit-dask-cudf.xml" -v --cov-config=.coveragerc --cov=dask_cudf --cov-report=xml:"$WORKSPACE/python/dask_cudf/dask-cudf-coverage.xml" --cov-report term dask_cudf

cd "$WORKSPACE/python/custreamz"
gpuci_logger "Python py.test for cuStreamz"
py.test -n 8 --cache-clear --basetemp="$WORKSPACE/custreamz-cuda-tmp" --junitxml="$WORKSPACE/junit-custreamz.xml" -v --cov-config=.coveragerc --cov=custreamz --cov-report=xml:"$WORKSPACE/python/custreamz/custreamz-coverage.xml" --cov-report term custreamz

gpuci_logger "Test notebooks"
"$WORKSPACE/ci/gpu/test-notebooks.sh" 2>&1 | tee nbtest.log
python "$WORKSPACE/ci/utils/nbtestlog2junitxml.py" nbtest.log

if [ -n "${CODECOV_TOKEN}" ]; then
    codecov -t $CODECOV_TOKEN
fi

return ${EXITCODE}
