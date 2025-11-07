#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

# This script runs compute-sanitizer on a single libcudf test executable
# Usage: ./run_single_memcheck.sh TEST_NAME [additional gtest args...]
# Example: ./run_single_memcheck.sh COLUMN_TEST --gtest_filter=ColumnTest.*

if [ $# -lt 1 ]; then
  echo "Error: Test name required"
  echo "Usage: $0 TEST_NAME [additional gtest args...]"
  exit 1
fi

TEST_NAME="$1"
shift

rapids-logger "Installing libcudf and libcudf-tests from rapidsai-nightly"

# Install packages from rapidsai-nightly channel
rapids-mamba-retry create -y -n libcudf -c rapidsai-nightly -c conda-forge libcudf libcudf-tests

rapids-logger "Running compute-sanitizer on $TEST_NAME"

export GTEST_CUDF_RMM_MODE=cuda
export GTEST_BRIEF=1
# compute-sanitizer bug 4553815
export LIBCUDF_MEMCHECK_ENABLED=1

# Navigate to test installation directory
TEST_DIR="${CONDA_PREFIX}/bin/gtests/libcudf"
TEST_EXECUTABLE="$TEST_DIR/$TEST_NAME"

if [ ! -x "$TEST_EXECUTABLE" ]; then
  rapids-logger "Error: Test executable $TEST_EXECUTABLE not found or not executable"
  exit 1
fi

rapids-logger "Check GPU usage"
nvidia-smi

# Run compute-sanitizer on the specified test
compute-sanitizer --tool memcheck "$TEST_EXECUTABLE" "$@"

EXITCODE=$?

# Clean up environment variables
unset GTEST_BRIEF
unset GTEST_CUDF_RMM_MODE
unset LIBCUDF_MEMCHECK_ENABLED

rapids-logger "compute-sanitizer on $TEST_NAME exiting with value: $EXITCODE"
exit $EXITCODE
