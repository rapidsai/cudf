#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

# Support customizing the ctests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest"

if [[ -d "${installed_test_location}" ]]; then
    cd "${installed_test_location}" || return
elif [[ -d "${devcontainers_test_location}" ]]; then
    cd "${devcontainers_test_location}" || return
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

ctest --output-on-failure --no-tests=error "$@"

./BITMASK_TEST --gtest_filter=CountBitmaskTest.IndexOfFirstUnsetBit

compute-sanitizer ./BITMASK_TEST --gtest_filter=CountBitmaskTest.IndexOfFirstUnsetBit --rmm_mode=cuda
