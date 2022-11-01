#!/bin/bash

# Copyright (c) 2019-2022, NVIDIA CORPORATION.

# cuDF build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)
# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean libcudf cudf cudfjar dask_cudf benchmarks tests libcudf_kafka cudf_kafka custreamz strings_udf -v -g -n -l --allgpuarch --disable_nvtx --opensource_nvcomp  --show_depr_warn --ptds -h --build_metrics --incl_cache_stats"
HELP="$0 [clean] [libcudf] [cudf] [cudfjar] [dask_cudf] [benchmarks] [tests] [libcudf_kafka] [cudf_kafka] [custreamz] [-v] [-g] [-n] [-h] [--cmake-args=\\\"<args>\\\"]
   clean                         - remove all existing build artifacts and configuration (start
                                   over)
   libcudf                       - build the cudf C++ code only
   cudf                          - build the cudf Python package
   cudfjar                       - build cudf JAR with static libcudf using devtoolset toolchain
   dask_cudf                     - build the dask_cudf Python package
   benchmarks                    - build benchmarks
   tests                         - build tests
   libcudf_kafka                 - build the libcudf_kafka C++ code only
   cudf_kafka                    - build the cudf_kafka Python package
   custreamz                     - build the custreamz Python package
   -v                            - verbose build mode
   -g                            - build for debug
   -n                            - no install step
   --allgpuarch                  - build for all supported GPU architectures
   --disable_nvtx                - disable inserting NVTX profiling ranges
   --opensource_nvcomp           - disable use of proprietary nvcomp extensions
   --show_depr_warn              - show cmake deprecation warnings
   --ptds                        - enable per-thread default stream
   --build_metrics               - generate build metrics report for libcudf
   --incl_cache_stats            - include cache statistics in build metrics report
   --cmake-args=\\\"<args>\\\"   - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   -h | --h[elp]                 - print this text

   default action (no args) is to build and install 'libcudf' then 'cudf'
   then 'dask_cudf' targets
"
LIB_BUILD_DIR=${LIB_BUILD_DIR:=${REPODIR}/cpp/build}
KAFKA_LIB_BUILD_DIR=${KAFKA_LIB_BUILD_DIR:=${REPODIR}/cpp/libcudf_kafka/build}
CUDF_KAFKA_BUILD_DIR=${REPODIR}/python/cudf_kafka/build
CUDF_BUILD_DIR=${REPODIR}/python/cudf/build
DASK_CUDF_BUILD_DIR=${REPODIR}/python/dask_cudf/build
CUSTREAMZ_BUILD_DIR=${REPODIR}/python/custreamz/build
CUDF_JAR_JAVA_BUILD_DIR="$REPODIR/java/target"

BUILD_DIRS="${LIB_BUILD_DIR} ${CUDF_BUILD_DIR} ${DASK_CUDF_BUILD_DIR} ${KAFKA_LIB_BUILD_DIR} ${CUDF_KAFKA_BUILD_DIR} ${CUSTREAMZ_BUILD_DIR} ${CUDF_JAR_JAVA_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_BENCHMARKS=OFF
BUILD_ALL_GPU_ARCH=0
BUILD_NVTX=ON
BUILD_TESTS=OFF
BUILD_DISABLE_DEPRECATION_WARNINGS=ON
BUILD_PER_THREAD_DEFAULT_STREAM=OFF
BUILD_REPORT_METRICS=OFF
BUILD_REPORT_INCL_CACHE_STATS=OFF
USE_PROPRIETARY_NVCOMP=ON

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo $ARGS | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo $EXTRA_CMAKE_ARGS | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
}

function buildAll {
    ((${NUMARGS} == 0 )) || !(echo " ${ARGS} " | grep -q " [^-]\+ ")
}

function buildLibCudfJniInDocker {
    local cudaVersion="11.5.0"
    local imageName="cudf-build:${cudaVersion}-devel-centos7"
    local CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
    local workspaceDir="/rapids"
    local localMavenRepo=${LOCAL_MAVEN_REPO:-"$HOME/.m2/repository"}
    local workspaceRepoDir="$workspaceDir/cudf"
    local workspaceMavenRepoDir="$workspaceDir/.m2/repository"
    local workspaceCcacheDir="$workspaceDir/.ccache"
    mkdir -p "$CUDF_JAR_JAVA_BUILD_DIR/libcudf-cmake-build"
    mkdir -p "$HOME/.ccache" "$HOME/.m2"
    nvidia-docker build \
        -f java/ci/Dockerfile.centos7 \
        --build-arg CUDA_VERSION=${cudaVersion} \
        -t $imageName .
    nvidia-docker run -it -u $(id -u):$(id -g) --rm \
        -e PARALLEL_LEVEL \
        -e CCACHE_DISABLE \
        -e CCACHE_DIR="$workspaceCcacheDir" \
        -v "/etc/group:/etc/group:ro" \
        -v "/etc/passwd:/etc/passwd:ro" \
        -v "/etc/shadow:/etc/shadow:ro" \
        -v "/etc/sudoers.d:/etc/sudoers.d:ro" \
        -v "$HOME/.ccache:$workspaceCcacheDir:rw" \
        -v "$REPODIR:$workspaceRepoDir:rw" \
        -v "$localMavenRepo:$workspaceMavenRepoDir:rw" \
        --workdir "$workspaceRepoDir/java/target/libcudf-cmake-build" \
        ${imageName} \
        scl enable devtoolset-9 \
            "cmake $workspaceRepoDir/cpp \
                -G${CMAKE_GENERATOR} \
                -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_LINKER_LAUNCHER=ccache \
                -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
                -DCUDA_STATIC_RUNTIME=ON \
                -DCMAKE_CUDA_ARCHITECTURES=${CUDF_CMAKE_CUDA_ARCHITECTURES} \
                -DCMAKE_INSTALL_PREFIX=/usr/local/rapids \
                -DUSE_NVTX=ON \
                -DCUDF_USE_PROPRIETARY_NVCOMP=ON \
                -DCUDF_USE_ARROW_STATIC=ON \
                -DCUDF_ENABLE_ARROW_S3=OFF \
                -DBUILD_TESTS=OFF \
                -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON \
                -DRMM_LOGGING_LEVEL=OFF \
                -DBUILD_SHARED_LIBS=OFF && \
             cmake --build . --parallel ${PARALLEL_LEVEL} && \
             cd $workspaceRepoDir/java && \
             mvn ${MVN_PHASES:-"package"} \
                -Dmaven.repo.local=$workspaceMavenRepoDir \
                -DskipTests=${SKIP_TESTS:-false} \
                -Dparallel.level=${PARALLEL_LEVEL} \
                -Dcmake.ccache.opts='-DCMAKE_C_COMPILER_LAUNCHER=ccache \
                                     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                     -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
                                     -DCMAKE_CXX_LINKER_LAUNCHER=ccache' \
                -DCUDF_CPP_BUILD_DIR=$workspaceRepoDir/java/target/libcudf-cmake-build \
                -DCUDA_STATIC_RUNTIME=ON \
                -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON \
                -DUSE_GDS=ON \
                -DGPU_ARCHS=${CUDF_CMAKE_CUDA_ARCHITECTURES} \
                -DCUDF_JNI_LIBCUDF_STATIC=ON \
                -Dtest=*,!CuFileTest,!CudaFatalTest"
}

if hasArg -h || hasArg --h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG="-v"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
    LIBCUDF_BUILD_DIR=${LIB_BUILD_DIR}
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg benchmarks; then
    BUILD_BENCHMARKS=ON
fi
if hasArg tests; then
    BUILD_TESTS=ON
fi
if hasArg --disable_nvtx; then
    BUILD_NVTX="OFF"
fi
if hasArg --opensource_nvcomp; then
    USE_PROPRIETARY_NVCOMP="OFF"
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNINGS=OFF
fi
if hasArg --ptds; then
    BUILD_PER_THREAD_DEFAULT_STREAM=ON
fi
if hasArg --build_metrics; then
    BUILD_REPORT_METRICS=ON
fi

if hasArg --incl_cache_stats; then
    BUILD_REPORT_INCL_CACHE_STATS=ON
fi

# Append `-DFIND_CUDF_CPP=ON` to EXTRA_CMAKE_ARGS unless a user specified the option.
if [[ "${EXTRA_CMAKE_ARGS}" != *"DFIND_CUDF_CPP"* ]]; then
    EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DFIND_CUDF_CPP=ON"
fi


# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
    if [ -d ${bd} ]; then
        find ${bd} -mindepth 1 -delete
        rmdir ${bd} || true
    fi
    done

    # Cleaning up python artifacts
    find ${REPODIR}/python/ | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild$)"  | xargs rm -rf

fi


################################################################################
# Configure, build, and install libcudf

if buildAll || hasArg libcudf || hasArg cudfjar; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        CUDF_CMAKE_CUDA_ARCHITECTURES="${CUDF_CMAKE_CUDA_ARCHITECTURES:-NATIVE}"
        if [[ "$CUDF_CMAKE_CUDA_ARCHITECTURES" == "NATIVE" ]]; then
            echo "Building for the architecture of the GPU in the system..."
        else
            echo "Building for the GPU architecture(s) $CUDF_CMAKE_CUDA_ARCHITECTURES ..."
        fi
    else
        CUDF_CMAKE_CUDA_ARCHITECTURES="ALL"
        echo "Building for *ALL* supported GPU architectures..."
    fi
fi

if buildAll || hasArg libcudf; then
    # get the current count before the compile starts
    if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" && -x "$(command -v sccache)" ]]; then
        # zero the sccache statistics
        sccache --zero-stats
    fi

    cmake -S $REPODIR/cpp -B ${LIB_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUDF_CMAKE_CUDA_ARCHITECTURES} \
          -DUSE_NVTX=${BUILD_NVTX} \
          -DCUDF_USE_PROPRIETARY_NVCOMP=${USE_PROPRIETARY_NVCOMP} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} \
          -DDISABLE_DEPRECATION_WARNINGS=${BUILD_DISABLE_DEPRECATION_WARNINGS} \
          -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=${BUILD_PER_THREAD_DEFAULT_STREAM} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          ${EXTRA_CMAKE_ARGS}

    cd ${LIB_BUILD_DIR}

    compile_start=$(date +%s)
    cmake --build . -j${PARALLEL_LEVEL} ${VERBOSE_FLAG}
    compile_end=$(date +%s)
    compile_total=$(( compile_end - compile_start ))

    # Record build times
    if [[ "$BUILD_REPORT_METRICS" == "ON" && -f "${LIB_BUILD_DIR}/.ninja_log" ]]; then
        echo "Formatting build metrics"
        python ${REPODIR}/cpp/scripts/sort_ninja_log.py ${LIB_BUILD_DIR}/.ninja_log --fmt xml > ${LIB_BUILD_DIR}/ninja_log.xml
        MSG="<p>"
        # get some sccache stats after the compile
        if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" && -x "$(command -v sccache)" ]]; then
           COMPILE_REQUESTS=$(sccache -s | grep "Compile requests \+ [0-9]\+$" | awk '{ print $NF }')
           CACHE_HITS=$(sccache -s | grep "Cache hits \+ [0-9]\+$" | awk '{ print $NF }')
           HIT_RATE=$(echo - | awk "{printf \"%.2f\n\", $CACHE_HITS / $COMPILE_REQUESTS * 100}")
           MSG="${MSG}<br/>cache hit rate ${HIT_RATE} %"
        fi
        MSG="${MSG}<br/>parallel setting: $PARALLEL_LEVEL"
        MSG="${MSG}<br/>parallel build time: $compile_total seconds"
        if [[ -f "${LIB_BUILD_DIR}/libcudf.so" ]]; then
           LIBCUDF_FS=$(ls -lh ${LIB_BUILD_DIR}/libcudf.so | awk '{print $5}')
           MSG="${MSG}<br/>libcudf.so size: $LIBCUDF_FS"
        fi
        echo "$MSG"
        python ${REPODIR}/cpp/scripts/sort_ninja_log.py ${LIB_BUILD_DIR}/.ninja_log --fmt html --msg "$MSG" > ${LIB_BUILD_DIR}/ninja_log.html
        cp ${LIB_BUILD_DIR}/.ninja_log ${LIB_BUILD_DIR}/ninja.log
    fi

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j${PARALLEL_LEVEL} --target install ${VERBOSE_FLAG}
    fi
fi

# Build and install the cudf Python package
if buildAll || hasArg cudf; then

    cd ${REPODIR}/python/cudf
    python setup.py build_ext --inplace -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_LIBRARY_PATH=${LIBCUDF_BUILD_DIR} -DCMAKE_CUDA_ARCHITECTURES=${CUDF_CMAKE_CUDA_ARCHITECTURES} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py install --single-version-externally-managed --record=record.txt  -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_LIBRARY_PATH=${LIBCUDF_BUILD_DIR} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    fi
fi

if buildAll || hasArg strings_udf; then

    cd ${REPODIR}/python/strings_udf
    python setup.py build_ext --inplace -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_LIBRARY_PATH=${LIBCUDF_BUILD_DIR} -DCMAKE_CUDA_ARCHITECTURES=${CUDF_CMAKE_CUDA_ARCHITECTURES} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py install --single-version-externally-managed --record=record.txt  -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_LIBRARY_PATH=${LIBCUDF_BUILD_DIR} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    fi
fi

# Build and install the dask_cudf Python package
if buildAll || hasArg dask_cudf; then

    cd ${REPODIR}/python/dask_cudf
    if [[ ${INSTALL_TARGET} != "" ]]; then
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace -j${PARALLEL_LEVEL}
        python setup.py install --single-version-externally-managed --record=record.txt
    else
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace -j${PARALLEL_LEVEL}
    fi
fi

if hasArg cudfjar; then
    buildLibCudfJniInDocker
fi

# Build libcudf_kafka library
if hasArg libcudf_kafka; then
    cmake -S $REPODIR/cpp/libcudf_kafka -B ${KAFKA_LIB_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          ${EXTRA_CMAKE_ARGS}


    cd ${KAFKA_LIB_BUILD_DIR}
    cmake --build . -j${PARALLEL_LEVEL} ${VERBOSE_FLAG}

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j${PARALLEL_LEVEL} --target install ${VERBOSE_FLAG}
    fi
fi

# build cudf_kafka Python package
if hasArg cudf_kafka; then
    cd ${REPODIR}/python/cudf_kafka
    if [[ ${INSTALL_TARGET} != "" ]]; then
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace -j${PARALLEL_LEVEL}
        python setup.py install --single-version-externally-managed --record=record.txt
    else
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace -j${PARALLEL_LEVEL} --library-dir=${LIBCUDF_BUILD_DIR}
    fi
fi

# build custreamz Python package
if hasArg custreamz; then
    cd ${REPODIR}/python/custreamz
    if [[ ${INSTALL_TARGET} != "" ]]; then
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace -j${PARALLEL_LEVEL}
        python setup.py install --single-version-externally-managed --record=record.txt
    else
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace -j${PARALLEL_LEVEL} --library-dir=${LIBCUDF_BUILD_DIR}
    fi
fi
