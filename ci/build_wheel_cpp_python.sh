#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build the C++ wheels first, then reuse both local wheelhouses while building
# the Python wheels. build_wheel_common.sh is idempotent within this shell, so
# sourcing the package-stage scripts does the common setup only once.
# shellcheck source=ci/build_wheel_cpp.sh
source ./ci/build_wheel_cpp.sh

# shellcheck source=ci/build_wheel_python.sh
source ./ci/build_wheel_python.sh
