#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# It is essential to cd into python/custreamz/custreamz/ as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_custreamz_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/custreamz/custreamz/

pytest --cache-clear "$@" tests
