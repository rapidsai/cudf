#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/pylibcudf"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Downloads libcudf wheel from this current build,
# then ensures 'pylibcudf' wheel builds always use the 'libcudf' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcudf_dist
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcudf_dist/libcudf_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

./ci/build_wheel.sh pylibcudf ${package_dir}

python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude libnvcomp.so \
    --exclude libkvikio.so \
    -w ${package_dir}/final_dist \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} final_dist

RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python ${package_dir}/final_dist
