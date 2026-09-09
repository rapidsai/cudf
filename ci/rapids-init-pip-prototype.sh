#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use the unreleased rapids-init-pip change under evaluation. Keeping the tool in
# the workspace makes it available to every pip invocation in this CI job, while
# deliberately not persisting it across jobs.
RAPIDS_INIT_PIP_SHA=a1aa4d1a80fcbf09da0f0b8eebf0e41ab42c0f48
RAPIDS_INIT_PIP_DIR="${GITHUB_WORKSPACE:-${PWD}}/.cache/gha-tools/${RAPIDS_INIT_PIP_SHA}"
RAPIDS_INIT_PIP_PATH="${RAPIDS_INIT_PIP_DIR}/rapids-init-pip"

if [[ ! -f "${RAPIDS_INIT_PIP_PATH}" ]]; then
  mkdir -p "${RAPIDS_INIT_PIP_DIR}"
  curl --fail --location --silent --show-error \
    "https://raw.githubusercontent.com/rapidsai/gha-tools/${RAPIDS_INIT_PIP_SHA}/tools/rapids-init-pip" \
    --output "${RAPIDS_INIT_PIP_PATH}"
fi

# shellcheck disable=SC1090
source "${RAPIDS_INIT_PIP_PATH}"
python -m pip cache dir
