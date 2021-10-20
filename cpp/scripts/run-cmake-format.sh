#!/bin/bash

# This script invokes cmake-format 
DEFAULT_FORMAT_FILE=cpp/build/release/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json
RAPIDS_CMAKE_FORMAT_FILE=${RAPIDS_CMAKE_FORMAT_FILE:-${DEFAULT_FORMAT_FILE}}

if [ -f ${RAPIDS_CMAKE_FORMAT_FILE} ]; then
  if [[ $1 == "cmake-format" ]]; then
    cmake-format -i --config-files cpp/cmake/config.json ${RAPIDS_CMAKE_FORMAT_FILE} -- ${@:2}
  elif [[ $1 == "cmake-lint" ]]; then
    cmake-lint --config-files cpp/cmake/config.json ${RAPIDS_CMAKE_FORMAT_FILE} -- ${@:2}
  fi
else
  echo "The rapids-cmake cmake-format configuration file is not present at ${RAPIDS_CMAKE_FORMAT_FILE}. Try setting the environment variable RAPIDS_CMAKE_FORMAT_FILE to the path to the config file."
fi
