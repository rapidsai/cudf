#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="cudf"
package_dir="python/cudf"

export SKBUILD_CMAKE_ARGS="-DUSE_LIBARROW_FROM_PYARROW=ON"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Downloads libcudf wheel from this current build,
# then ensures 'cudf' wheel builds always use the 'libcudf' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcudf_dist

echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcudf_dist/libcudf_*.whl)" > /tmp/constraints.txt

# --- start of section to remove ---#
# TODO: remove this before merging
# use librmm and rmm from
RAPIDS_REPOSITORY=rmm \
RAPIDS_BUILD_TYPE=pull-requst \
RAPIDS_REF_NAME=1644 \
RAPIDS_SHA=0701559 \
RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-download-wheels-from-s3 cpp /tmp/libcudf_dist

RAPIDS_REPOSITORY=rmm \
RAPIDS_BUILD_TYPE=pull-requst \
RAPIDS_REF_NAME=1644 \
RAPIDS_SHA=0701559 \
RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-download-wheels-from-s3 python /tmp/libcudf_dist

echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcudf_dist/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcudf_dist/librmm_*.whl)" >> /tmp/constraints.txt
# --- end of section to remove ---#

export PIP_CONSTRAINT="/tmp/constraints.txt"
./ci/build_wheel.sh ${package_dir}

cd ${package_dir}
mkdir -p final_dist
python -m auditwheel repair --exclude libcudf.so --exclude libarrow.so.1601 -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist
