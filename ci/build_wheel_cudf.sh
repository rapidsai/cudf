#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cudf"

export SKBUILD_CONFIGURE_OPTIONS="-DCUDF_BUILD_WHEELS=ON -DDETECT_CONDA_ENV=OFF"

# Force a build using the latest version of the code before this PR
CUDF_BUILD_BRANCH=${1}
WHEEL_NAME_PREFIX="cudf_"
if [[ "${CUDF_BUILD_BRANCH}" == "main" ]]; then
    git checkout branch-23.10-xdf
    WHEEL_NAME_PREFIX="cudf_${CUDF_BUILD_BRANCH}_"
fi

./ci/build_wheel.sh cudf ${package_dir}

mkdir -p ${package_dir}/final_dist
python -m auditwheel repair -w ${package_dir}/final_dist ${package_dir}/dist/*


RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${WHEEL_NAME_PREFIX}${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
