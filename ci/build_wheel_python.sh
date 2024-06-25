#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="cudf"
package_dir="python/cudf"

export SKBUILD_CMAKE_ARGS="-DUSE_LIBARROW_FROM_PYARROW=ON"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
# Downloads libcudf wheel from this current build, then points pip to it in PIP_FIND_LINKS below
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcudf_dist

export PIP_FIND_LINKS="/tmp/libcudf_dist"
./ci/build_wheel.sh ${package_dir}

cd ${package_dir}
mkdir -p final_dist
python -m auditwheel repair --exclude libcudf.so --exclude libarrow.so.1601 -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist
