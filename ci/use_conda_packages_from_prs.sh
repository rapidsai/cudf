#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Download conda CI artifacts from dependent PRs

LIBRMM_CHANNEL=$(rapids-get-pr-artifact rmm 2106 cpp conda)
RMM_CHANNEL=$(rapids-get-pr-artifact rmm 2106 python conda)

RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBRMM_CHANNEL}"
    "${RMM_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"
do
   conda config --system --add channels "${_channel}"
done
