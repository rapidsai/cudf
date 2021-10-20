#!/bin/bash

# This script is a pre-commit hook that wraps cmakelang's cmake linters. The
# wrapping is necessary because RAPIDS libraries split configuration for
# cmakelang linters between a local config file and a second config file that's
# shared across all of RAPIDS via rapids-cmake. In order to keep it up to date
# this script is only maintained in one place (the rapids-cmake repo) and
# pulled down during builds. We need a way to invoke CMake linting commands
# without causing pre-commit failures (which could block local commits or CI),
# while also being sufficiently flexible to allow users to maintain the config
# file independently of a build directory. This script provides the minimal
# functionality to enable those use cases. While this script can be invoked
# directly, it is advisable to instead use the pre-commit hooks via
# `pre-commit run (cmake-format)|(cmake-format)`
#
# Usage:
# bash run-cmake-format.sh {cmake-format,cmake-lint} infile [infile ...]

DEFAULT_FORMAT_FILE=cpp/build/release/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json
RAPIDS_CMAKE_FORMAT_FILE=${RAPIDS_CMAKE_FORMAT_FILE:-${DEFAULT_FORMAT_FILE}}

if [ -f ${RAPIDS_CMAKE_FORMAT_FILE} ]; then
  if [[ $1 == "cmake-format" ]]; then
    cmake-format -i --config-files cpp/cmake/config.json ${RAPIDS_CMAKE_FORMAT_FILE} -- ${@:2}
  elif [[ $1 == "cmake-lint" ]]; then
    cmake-lint --config-files cpp/cmake/config.json ${RAPIDS_CMAKE_FORMAT_FILE} -- ${@:2}
  fi
else
  echo "The rapids-cmake cmake-format configuration file is not present at ${RAPIDS_CMAKE_FORMAT_FILE}."\
       "Try setting the environment variable RAPIDS_CMAKE_FORMAT_FILE to the path to the config file."
fi
