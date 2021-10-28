#!/bin/bash

# This script is a pre-commit hook that wraps cmakelang's cmake linters. The
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
# While this script can be invoked directly (but only from the repo root since
# all paths are relative to that), it is advisable to instead use the
# pre-commit hooks via
# `pre-commit run (cmake-format)|(cmake-format)`.
#
# Usage:
# bash run-cmake-format.sh {cmake-format,cmake-lint} infile [infile ...]

# Note that pre-commit always runs from the root of the repository, so relative
# paths are automatically relative to the repo root.
DEFAULT_FORMAT_FILE_LOCATIONS=(
  "cpp/build/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json" 
  "${CUDF_ROOT:-${HOME}}/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json"
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
  cmake-lint --config-files cpp/cmake/config.json ${RAPIDS_CMAKE_FORMAT_FILE} -- ${@:2}
fi
