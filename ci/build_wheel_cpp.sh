#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/libcudf"

# --- start of section to remove ---#
# TODO: remove this before merging

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# use librmm and rmm from https://github.com/rapidsai/rmm/pull/1644
RAPIDS_REPOSITORY=rmm \
RAPIDS_BUILD_TYPE=pull-request \
RAPIDS_REF_NAME=1644 \
RAPIDS_SHA=e93f26c \
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-download-wheels-from-s3 cpp /tmp/libcudf_dist

RAPIDS_REPOSITORY=rmm \
RAPIDS_BUILD_TYPE=pull-request \
RAPIDS_REF_NAME=1644 \
RAPIDS_SHA=e93f26c \
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-download-wheels-from-s3 python /tmp/libcudf_dist

echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcudf_dist/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcudf_dist/rmm_*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT=/tmp/constraints.txt
# --- end of section to remove ---#

./ci/build_wheel.sh ${package_dir}

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

mkdir -p ${package_dir}/final_dist
python -m auditwheel repair --exclude libarrow.so.1601 -w ${package_dir}/final_dist ${package_dir}/dist/*

RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp ${package_dir}/final_dist
