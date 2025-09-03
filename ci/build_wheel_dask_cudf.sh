#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

package_dir="python/dask_cudf"

./ci/build_wheel.sh dask-cudf ${package_dir}
cp "${package_dir}/dist"/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
