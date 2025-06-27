#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source ci/use_gha_tools_from_branch.sh
source rapids-init-pip
source ci/use_wheels_from_prs.sh

package_dir="python/cudf_polars"

./ci/build_wheel.sh cudf-polars ${package_dir}
cp "${package_dir}/dist"/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
