#!/bin/bash

# Copyright (c) 2021-2024, NVIDIA CORPORATION.

# libcudf examples build script

set -euo pipefail

# Parallelism control
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Root of examples
EXAMPLES_DIR=$(dirname "$(realpath "$0")")
LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${EXAMPLES_DIR}/../build")}

################################################################################
# Add individual libcudf examples build scripts down below

build_example() {
  example_dir=${1}
  example_dir="${EXAMPLES_DIR}/${example_dir}"
  build_dir="${example_dir}/build"

  # Configure
  cmake -S ${example_dir} -B ${build_dir} -Dcudf_ROOT="${LIB_BUILD_DIR}"
  # Build
  cmake --build ${build_dir} -j${PARALLEL_LEVEL}
}

build_example basic
build_example strings
build_example nested_types
