#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

rapids-generate-version > ./VERSION
rapids-generate-version > ./python/cudf/cudf/VERSION

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
rapids-telemetry-record build-${package_name}.log rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

rapids-telemetry-record sccache-stats-${package_name}.txt sccache --show-adv-stats
