#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cudf_polars"

./ci/build_wheel.sh cudf-polars ${package_dir}
./ci/validate_wheel.sh ${package_dir} dist

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_polars_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 python ${package_dir}/dist
