#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cudf"
wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR:-"${package_dir}/final_dist"}

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcudf and pylibcudf wheels from this current build,
# then ensures 'cudf' wheel builds always use the 'libcudf' and 'pylibcudf' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp)
PYTHON_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python)
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${CPP_WHEELHOUSE}/libcudf_*.whl)" > /tmp/constraints.txt
echo "pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYTHON_WHEELHOUSE}/pylibcudf_*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

./ci/build_wheel.sh cudf ${package_dir}

python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude libnvcomp.so \
    --exclude libkvikio.so \
    --exclude librapids_logger.so \
    -w ${wheel_dir} \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${wheel_dir}

# RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python ${package_dir}/final_dist
