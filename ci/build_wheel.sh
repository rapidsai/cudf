#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
initial_wheel_dir=$3

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
rapids-pip-retry wheel \
    -w "${initial_wheel_dir}" \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats
