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
pushd $(dirname $0)
REPODIR=$(pwd)
popd

VALIDARGS="libcudf cudf clean -v -g -h"
HELP="$0 [cudf | libcudf | clean] [-v] [-g] [-h]
   libcudf - build the cudf C++ code only
   cudf    - build the cudf Python package
   clean   - remove all existing build artifacts and configuration (start over)
   -v      - verbose build mode
   -g      - build for debug
   -h      - print this text
"
LIBCUDF_BUILD_DIR=${REPODIR}/cpp/build
CUDF_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${LIBCUDF_BUILD_DIR} ${CUDF_BUILD_DIR}"
VERBOSE=0
BUILD_TYPE=Release

# Set defaults for vars that may not have been definied externally
# NOTE: default PREFIX assumes a conda environment is being used!
PYTHON=${PYTHON:=python}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}
PREFIX=${PREFIX:=${CONDA_PREFIX}}

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

# Set flags accordingly
if hasArg -v; then
    VERBOSE=1
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    find ${BUILD_DIRS} -mindepth 1 -delete
    rmdir ${BUILD_DIRS} || true
fi

################################################################################
# Configure and build libcudf
if (( ${NUMARGS} == 0 )) || hasArg libcudf; then

    mkdir -p ${LIBCUDF_BUILD_DIR}
    cd ${LIBCUDF_BUILD_DIR}

    cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} \
          -DCMAKE_CXX11_ABI=ON \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} install
fi

# Build the cudf Python package
if (( ${NUMARGS} == 0 )) || hasArg cudf; then

    cd ${REPODIR}/python
    ${PYTHON} setup.py build_ext --inplace
    ${PYTHON} setup.py install --single-version-externally-managed --record=record.txt
fi

