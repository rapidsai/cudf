# !/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# download CI artifacts
LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1909 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1909 python)

# make sure they can be found locally
conda config --system --add channels "${LIBRMM_CHANNEL}"
conda config --system --add channels "${RMM_CHANNEL}"
