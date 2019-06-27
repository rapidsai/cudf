#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.

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

VALIDARGS="clean libcudf cudf -v -g -n -h"
HELP="$0 [clean] [libcudf] [cudf] [-v] [-g] [-n] [-h]
   clean   - remove all existing build artifacts and configuration (start over)
   libcudf - build the cudf C++ code only
   cudf    - build the cudf Python package
   -v      - verbose build mode
   -g      - build for debug
   -n      - no install step
   -h      - print this text

   default action (no args) is to build and install 'libcudf' then 'cudf' targets
"
LIBCUDF_BUILD_DIR=${REPODIR}/cpp/build
CUDF_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${LIBCUDF_BUILD_DIR} ${CUDF_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
	if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
	    echo "Invalid option: ${a}"
	    exit 1
	fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE=1
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
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
fi

################################################################################
# Configure, build, and install libcudf
if (( ${NUMARGS} == 0 )) || hasArg libcudf; then

    mkdir -p ${LIBCUDF_BUILD_DIR}
    cd ${LIBCUDF_BUILD_DIR}
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CXX11_ABI=ON \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} ${INSTALL_TARGET}
fi

# Build and install the cudf Python package
if (( ${NUMARGS} == 0 )) || hasArg cudf; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
	python setup.py build_ext --inplace
	python setup.py install --single-version-externally-managed --record=record.txt
    else
	python setup.py build_ext --inplace --library-dir=${LIBCUDF_BUILD_DIR}
    fi
fi
