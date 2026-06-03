#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_cudf_streaming_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_streaming/cudf_streaming/tests

# OpenMPI specific options (CI runs as root)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1

# cudf_streaming tests require MPI for the communicator fixtures.
# Run with mpirun; currently single-rank only tests exist.
mpirun --map-by node --bind-to none -np 1 \
  python -m pytest --cache-clear "$@" .
