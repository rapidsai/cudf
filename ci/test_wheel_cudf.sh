#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

# Set the manylinux version to pull the newer ABI wheels when testing on Ubuntu 20.04 or later.
manylinux="manylinux_2_17"
if command -v lsb_release >/dev/null 2>&1 ; then
    release_info=$(lsb_release -r)
    major_version=$(echo "$release_info" | awk '{print $2}' | cut -d. -f1)
    if [[ ${major_version} -ge 20 ]]; then
        manylinux="manylinux_2_28"
    fi
fi

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_${manylinux}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cudf*.whl)[test]

# Run smoke tests for aarch64 pull requests
if [[ "$(arch)" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_cudf.py
else
    python -m pytest -n 8 ./python/cudf/cudf/tests
fi
