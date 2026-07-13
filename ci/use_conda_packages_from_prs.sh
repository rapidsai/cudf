#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# download CI artifacts
LIBRAPIDSMPF_CHANNEL=$(rapids-get-pr-artifact rapidsmpf 1106 cpp conda)
RAPIDSMPF_CHANNEL=$(rapids-get-pr-artifact rapidsmpf 1106 python conda)

# For `rattler` builds:
# Add these channels to the array checked by 'rapids-rattler-channel-string' so
# locally-downloaded packages are preferred under strict channel priority.
RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBRAPIDSMPF_CHANNEL}"
    "${RAPIDSMPF_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

# For tests and `conda-build` builds:
# Prepend to conda's system-wide channel list.
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"
do
   conda config --system --add channels "${_channel}"
done

# For `conda env create -f <env.yaml>` jobs (C++/Python/Java/notebook tests, docs):
# `conda env create` gives channels listed in the env file HIGHER priority than
# the system `.condarc`, so `conda config --system --add channels` is NOT enough:
# under strict channel priority the solver still prefers e.g. `rapidsai-nightly`
# listed in the generated env file. Expose the channels as `--prepend-channel`
# args so callers can inject them into the env file ahead of the defaults.
RAPIDS_PREPENDED_CHANNEL_ARGS=()
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"
do
   RAPIDS_PREPENDED_CHANNEL_ARGS+=(--prepend-channel "${_channel}")
done
export RAPIDS_PREPENDED_CHANNEL_ARGS
