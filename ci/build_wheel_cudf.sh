#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

if [[ ! -d "/tmp/gha-tools" ]]; then
  git clone https://github.com/divyegala/gha-tools.git -b wheel-local-runs /tmp/gha-tools
fi

source /tmp/gha-tools/tools/rapids-configure-sccache

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_EPOCH_TIMESTAMP})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

bash ci/release/apply_wheel_modifications.sh ${version_override} "-${RAPIDS_PY_CUDA_SUFFIX}"
echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

cd python/cudf

# Hardcode the output dir
SKBUILD_CONFIGURE_OPTIONS="-DCUDF_BUILD_WHEELS=ON -DDETECT_CONDA_ENV=OFF" python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" /tmp/gha-tools/tools/rapids-upload-wheels-to-s3 final_dist
