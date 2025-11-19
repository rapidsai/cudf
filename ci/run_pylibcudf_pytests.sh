#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/pylibcudf/pylibcudf as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/pylibcudf/

pytest --cache-clear --ignore="benchmarks" "$@" tests
