#!/bin/bash

# Copyright (c) 2021-2024, NVIDIA CORPORATION.

# libcudf examples build script

set -euo pipefail

# Parallelism control
PARALLEL_LEVEL=${PARALLEL_LEVEL:-8}
# Installation disabled by default
INSTALL_EXAMPLES=false

# Check for -i or --install flags to enable installation
ARGS=$(getopt -o i --long install -- "$@")
eval set -- "$ARGS"
while [ : ]; do
  case "$1" in
    -i | --install)
        INSTALL_EXAMPLES=true
        shift
        ;;
    --) shift;
        break
        ;;
  esac
done

# Root of examples
EXAMPLES_DIR=$(dirname "$(realpath "$0")")

# Set up default libcudf build directory and install prefix if conda build
if [ "${CONDA_BUILD:-"0"}" == "1" ]; then
  LIB_BUILD_DIR="${LIB_BUILD_DIR:-${SRC_DIR/cpp/build}}"
  INSTALL_PREFIX="${INSTALL_PREFIX:-${PREFIX}}"
fi

# libcudf build directory
LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${EXAMPLES_DIR}/../build")}

################################################################################
# Add individual libcudf examples build scripts down below

build_example() {
  example_dir=${1}
  example_dir="${EXAMPLES_DIR}/${example_dir}"
  build_dir="${example_dir}/build"

  # Configure
  cmake -S ${example_dir} -B ${build_dir} -Dcudf_ROOT="${LIB_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=86
  # Build
  cmake --build ${build_dir} -j${PARALLEL_LEVEL}
  # Install if needed
  if [ "$INSTALL_EXAMPLES" = true ]; then
    cmake --install ${build_dir} --prefix ${INSTALL_PREFIX:-${example_dir}/install}
  fi
}

# build_example basic
# build_example strings
# build_example nested_types
build_example parquet_io
build_example orc_io
# build_example billion_rows
# build_example interop
