#!/bin/bash
# Copyright (c) 2021-2024, NVIDIA CORPORATION.

# This script is a wrapper for cmakelang that may be used with pre-commit. The
# wrapping is necessary because RAPIDS libraries split configuration for
# cmakelang linters between a local config file and a second config file that's
# shared across all of RAPIDS via rapids-cmake. In order to keep it up to date
# this file is only maintained in one place (the rapids-cmake repo) and
# pulled down during builds. We need a way to invoke CMake linting commands
# without causing pre-commit failures (which could block local commits or CI),
# while also being sufficiently flexible to allow users to maintain the config
# file independently of a build directory.
#
# This script provides the minimal functionality to enable those use cases. It
# searches in a number of predefined locations for the rapids-cmake config file
# and exits gracefully if the file is not found. If a user wishes to specify a
# config file at a nonstandard location, they may do so by setting the
# environment variable RAPIDS_CMAKE_FORMAT_FILE.
#
# This script can be invoked directly anywhere within the project repository.
# Alternatively, it may be invoked as a pre-commit hook via
# `pre-commit run (cmake-format)|(cmake-lint)`.
#
# Usage:
# bash run-cmake-format.sh {cmake-format,cmake-lint} infile [infile ...]

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

DEFAULT_FORMAT_FILE_LOCATIONS=(
  "${CUDF_BUILD_DIR:-${HOME}}/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json"
  "${CUDF_BUILD_DIR:-cpp/build}/latest/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json"
  "cpp/libcudf_kafka/build/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json"
)

if [ -z ${RAPIDS_CMAKE_FORMAT_FILE:+PLACEHOLDER} ]; then
    for file_path in ${DEFAULT_FORMAT_FILE_LOCATIONS[@]}; do
        if [ -f ${file_path} ]; then
            RAPIDS_CMAKE_FORMAT_FILE=${file_path}
            break
        fi
    done
fi

if [ -z ${RAPIDS_CMAKE_FORMAT_FILE:+PLACEHOLDER} ]; then
  echo "The rapids-cmake cmake-format configuration file was not found at any of the default search locations: "
  echo ""
  ( IFS=$'\n'; echo "${DEFAULT_FORMAT_FILE_LOCATIONS[*]}" )
  echo ""
  echo "Try setting the environment variable RAPIDS_CMAKE_FORMAT_FILE to the path to the config file."
  exit 0
else
  echo "Using format file ${RAPIDS_CMAKE_FORMAT_FILE}"
fi

if [[ $1 == "cmake-format" ]]; then
  cmake-format -i --config-files cpp/cmake/config.json ${RAPIDS_CMAKE_FORMAT_FILE} -- ${@:2}
elif [[ $1 == "cmake-lint" ]]; then
  # Since the pre-commit hook is verbose, we have to be careful to only
  # present cmake-lint's output (which is quite verbose) if we actually
  # observe a failure.
  OUTPUT=$(cmake-lint --config-files cpp/cmake/config.json ${RAPIDS_CMAKE_FORMAT_FILE} -- ${@:2})
  status=$?

  if ! [ ${status} -eq 0 ]; then
    echo "${OUTPUT}"
  fi
  exit ${status}
fi
