#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cudf"

export SKBUILD_CONFIGURE_OPTIONS="-DCUDF_BUILD_WHEELS=ON -DDETECT_CONDA_ENV=OFF"

# Force a build using the latest version of the code before this PR
CUDF_BUILD_BRANCH=${1:-""}
WHEEL_NAME="cudf"
if [[ "${CUDF_BUILD_BRANCH}" == "main" ]]; then
    MAIN_COMMIT=$(git merge-base HEAD origin/branch-23.10-xdf)
    git checkout $MAIN_COMMIT
    WHEEL_NAME="${WHEEL_NAME}_${CUDF_BUILD_BRANCH}"
fi

./ci/build_wheel.sh ${WHEEL_NAME} ${package_dir}

mkdir -p ${package_dir}/final_dist
python -m auditwheel repair -w ${package_dir}/final_dist ${package_dir}/dist/*


RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${WHEEL_NAME}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
