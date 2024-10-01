#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# This script is a wrapper for clang-tidy for use with pre-commit. The
# wrapping serves a few purposes:
# 1. It allows us to find the files to run clang-tidy on rather than relying on pre-commit to pass them in.
# 2. It allows us to fail gracefully if no compile commands database is found.
#
# This script can be invoked directly anywhere within the project repository.
# Alternatively, it may be invoked as a pre-commit hook via
# `pre-commit run clang-tidy`.
#
# Usage:
# bash run-clang-tidy.sh

status=0
if [ -z ${CUDF_ROOT:+PLACEHOLDER} ]; then
    CUDF_BUILD_DIR=$(git rev-parse --show-toplevel 2>&1)/cpp/build
    status=$?
else
    CUDF_BUILD_DIR=${CUDF_ROOT}
fi

if ! [ ${status} -eq 0 ]; then
    if [[ ${CUDF_BUILD_DIR} == *"not a git repository"* ]]; then
        echo "This script must be run inside the cudf repository, or the CUDF_ROOT environment variable must be set."
    else
        echo "Script failed with unknown error attempting to determine project root:"
        echo ${CUDF_BUILD_DIR}
    fi
    exit 1
fi

if [ ! -f "${CUDF_BUILD_DIR}/compile_commands.json" ]; then
    echo "No compile commands database found. Please run CMake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON."
    exit 1
fi


find cpp/{src,tests} -name *.cpp | grep -v -E ".*(cpu_unbz2.cpp|brotli_dict.cpp).*" | xargs -n 1 -P 10 clang-tidy -p ${CUDF_BUILD_DIR} --extra-arg="-Qunused-arguments"
