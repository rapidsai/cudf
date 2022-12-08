#!/bin/bash

# Copyright (c) 2021-2022, NVIDIA CORPORATION.

# libcudf examples build script

# Parallelism control
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Root of examples
EXAMPLES_DIR=$(dirname "$(realpath "$0")")
LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${EXAMPLES_DIR}/../build")}

################################################################################
# Add individual libcudf examples build scripts down below

# Basic example
BASIC_EXAMPLE_DIR=${EXAMPLES_DIR}/basic
BASIC_EXAMPLE_BUILD_DIR=${BASIC_EXAMPLE_DIR}/build
# Configure
cmake -S ${BASIC_EXAMPLE_DIR} -B ${BASIC_EXAMPLE_BUILD_DIR} -Dcudf_ROOT="${LIB_BUILD_DIR}"
# Build
cmake --build ${BASIC_EXAMPLE_BUILD_DIR} -j${PARALLEL_LEVEL}

# Strings example
STRINGS_EXAMPLE_DIR=${EXAMPLES_DIR}/strings
STRINGS_EXAMPLE_BUILD_DIR=${STRINGS_EXAMPLE_DIR}/build
# Configure
cmake -S ${STRINGS_EXAMPLE_DIR} -B ${STRINGS_EXAMPLE_BUILD_DIR} -Dcudf_ROOT="${LIB_BUILD_DIR}"
# Build
cmake --build ${STRINGS_EXAMPLE_BUILD_DIR} -j${PARALLEL_LEVEL}
