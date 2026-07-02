#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# download CI artifacts
LIBRAPIDSMPF_CHANNEL=$(rapids-get-pr-artifact rapidsmpf 1081 cpp conda)
RAPIDSMPF_CHANNEL=$(rapids-get-pr-artifact rapidsmpf 1081 python conda)

# For `rattler` builds:
#
# Add these channels to the array checked by 'rapids-rattler-channel-string'.
# This ensures that when conda packages are built with strict channel priority enabled,
# the locally-downloaded packages will be preferred to remote packages (e.g. nightlies).
#
RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBRAPIDSMPF_CHANNEL}"
    "${RAPIDSMPF_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

# For tests and `conda-build` builds:
#
# Add these channels to the system-wide conda configuration.
# This results in PREPENDING them to conda's channel list, so
# these packages should be found first if strict channel priority is enabled.
#
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"
do
   conda config --system --add channels "${_channel}"
done

# For `conda env create -f <env.yaml>` jobs (e.g. C++/Python/Java/notebook
# tests, docs builds):
#
# `conda env create` gives the channels listed in the environment file a HIGHER
# priority than the channels in the system/user `.condarc`. So the
# `conda config --system --add channels` calls above are NOT sufficient on their
# own: with strict channel priority the solver still prefers e.g.
# `rapidsai-nightly` (which is listed in the generated env file) over these PR
# channels, pulling in the wrong build of `librapidsmpf`/`rapidsmpf`.
#
# Expose the channels as `--prepend-channel` arguments so callers can inject
# them into the generated environment file ahead of the default channels.
RAPIDS_PREPENDED_CHANNEL_ARGS=()
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"
do
   RAPIDS_PREPENDED_CHANNEL_ARGS+=(--prepend-channel "${_channel}")
done
export RAPIDS_PREPENDED_CHANNEL_ARGS
