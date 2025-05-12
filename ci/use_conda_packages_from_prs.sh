#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# download CI artifacts
LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1909 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1909 python)

# For `rattler` builds:
#
# Add these channels to the array checked by 'rapids-rattler-channel-string'.
# This ensures that when conda packages are built with strict channel priority enabled,
# the locally-downloaded packages will be preferred to remote packages (e.g. nightlies).
#
RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBRMM_CHANNEL}"
    "${RMM_CHANNEL}"
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
