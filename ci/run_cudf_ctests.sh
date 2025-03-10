#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf/";

export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8

#ctest --output-on-failure --no-tests=error "$@"
compute-sanitizer ./TEXT_TEST --gtest_filter=TextSubwordTest.Tokenize --rmm_mode=cuda
compute-sanitizer ./TEXT_TEST --gtest_filter=TextBytePairEncoding.BytePairEncoding --rmm_mode=cuda
