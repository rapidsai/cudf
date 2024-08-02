#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cudf"

export SKBUILD_CMAKE_ARGS="-DUSE_LIBARROW_FROM_PYARROW=ON"

# Download the pylibcudf built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 /tmp/pylibcudf_dist

echo "pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/pylibcudf_dist/pylibcudf_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"
./ci/build_wheel.sh ${package_dir}

python -m auditwheel repair -w ${package_dir}/final_dist ${package_dir}/dist/*

RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
