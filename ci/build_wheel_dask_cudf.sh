#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/dask_cudf"
wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR:-"${package_dir}/dist"}

./ci/build_wheel.sh dask-cudf ${wheel_dir}
./ci/validate_wheel.sh ${wheel_dir}

# RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
# RAPIDS_PY_WHEEL_NAME="dask_cudf_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 python ${package_dir}/dist
