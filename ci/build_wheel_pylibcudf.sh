#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source ci/use_gha_tools_from_branch.sh
source ci/use_wheels_from_prs.sh

package_dir="python/pylibcudf"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcudf wheel from this current build,
# then ensures 'pylibcudf' wheel builds always use the 'libcudf' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
LIBCUDF_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUDF_WHEELHOUSE}/libcudf_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

./ci/build_wheel.sh pylibcudf ${package_dir}

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude libnvcomp.so \
    --exclude libkvikio.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
