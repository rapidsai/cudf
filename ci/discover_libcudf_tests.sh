#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Installing libcudf and libcudf-tests from rapidsai-nightly"

# Install packages from rapidsai-nightly channel
rapids-mamba-retry create -y -n libcudf -c rapidsai-nightly -c conda-forge libcudf libcudf-tests

rapids-logger "Discovering libcudf test executables"

# Navigate to test installation directory
TEST_DIR="${CONDA_PREFIX}/bin/gtests/libcudf"

if [ ! -d "$TEST_DIR" ]; then
  rapids-logger "Error: Test directory $TEST_DIR not found"
  exit 1
fi

cd "$TEST_DIR"

# Find all *_TEST executables
if ! ls *_TEST 1> /dev/null 2>&1; then
  rapids-logger "Error: No test executables found matching *_TEST pattern"
  exit 1
fi

# Create JSON array of test names
tests=$(ls -1 *_TEST | jq -R -s -c 'split("\n") | map(select(length > 0))')

rapids-logger "Found tests: $tests"

# Output to GITHUB_OUTPUT if available (for GitHub Actions)
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "tests=$tests" >> "$GITHUB_OUTPUT"
fi

# Also print to stdout for direct script usage
echo "$tests"
