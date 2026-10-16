#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Resolve an exact version so an incompatible solve cannot silently retain the
# older rattler-build installed in the CI image.
RATTLER_BUILD_VERSION=$(
    conda search --json --override-channels -c conda-forge rattler-build |
        python -c '
import json
import sys
from packaging.version import Version

versions = [Version(record["version"]) for record in json.load(sys.stdin)["rattler-build"]]
print(max(version for version in versions if not version.is_prerelease and not version.is_devrelease))
'
)

rapids-conda-retry install -y -c conda-forge "rattler-build=${RATTLER_BUILD_VERSION}"

RATTLER_BUILD_INSTALLED_VERSION=$(rattler-build --version)
if [[ "${RATTLER_BUILD_INSTALLED_VERSION}" != "rattler-build ${RATTLER_BUILD_VERSION}" ]]; then
    echo "Expected rattler-build ${RATTLER_BUILD_VERSION}, found ${RATTLER_BUILD_INSTALLED_VERSION}" >&2
    exit 1
fi
echo "${RATTLER_BUILD_INSTALLED_VERSION}"
