#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/dask_cudf"

wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}

./ci/build_wheel.sh dask-cudf ${package_dir}
mkdir -p "${wheel_dir}"
cp "${package_dir}/dist"/* "${wheel_dir}/"
./ci/validate_wheel.sh "${package_dir}" "${wheel_dir}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 python "${wheel_dir}"
