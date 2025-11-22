#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}-${RAPIDS_CONDA_ARCH}-cuda${RAPIDS_CUDA_VERSION%%.*}-wheel-preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

rapids-generate-version > ./VERSION
rapids-generate-version > ./python/cudf/cudf/VERSION

cd "${package_dir}"

sccache --stop-server 2>/dev/null || true

rapids-logger "Building '${package_name}' wheel"
rapids-telemetry-record build-${package_name}.log rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

rapids-telemetry-record sccache-stats-${package_name}.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true
