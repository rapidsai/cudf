#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Temporary prototype for https://github.com/rapidsai/build-planning/issues/260.
# Download the unreleased rapids-init-pip implementation by immutable commit SHA,
# then let the existing test script use it unchanged.
set -eoxu pipefail

GHA_TOOLS_SHA="a1aa4d1a80fcbf09da0f0b8eebf0e41ab42c0f48"
WORKSPACE_ROOT="${GITHUB_WORKSPACE:-${PWD}}"
GHA_TOOLS_BIN="${WORKSPACE_ROOT}/.gha-tools-prototype/bin"

mkdir -p "${GHA_TOOLS_BIN}"
curl --fail --location --silent --show-error \
    "https://raw.githubusercontent.com/rapidsai/gha-tools/${GHA_TOOLS_SHA}/tools/rapids-init-pip" \
    --output "${GHA_TOOLS_BIN}/rapids-init-pip"
chmod +x "${GHA_TOOLS_BIN}/rapids-init-pip"
export PATH="${GHA_TOOLS_BIN}:${PATH}"

# Initialize the cache before reporting its state and before the test script
# sources the same tool for its normal setup.
source rapids-init-pip
python -m pip cache dir
python -m pip cache info

source ci/cudf_pandas_scripts/run_tests.sh

python -m pip cache info
