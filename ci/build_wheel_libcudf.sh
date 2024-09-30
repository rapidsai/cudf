#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/libcudf"

export SKBUILD_CMAKE_ARGS="-DUSE_CUDA_NVCOMP_WHEEL=ON"
./ci/build_wheel.sh ${package_dir}

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

mkdir -p ${package_dir}/final_dist
python -m auditwheel repair -w ${package_dir}/final_dist --exclude libnvcomp.so.4 ${package_dir}/dist/*

RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp ${package_dir}/final_dist
