#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cudf"

export SKBUILD_CONFIGURE_OPTIONS="-DCUDF_BUILD_WHEELS=ON -DDETECT_CONDA_ENV=OFF"

./ci/build_wheel.sh cudf ${package_dir}

manylinux="manylinux_2_17"
mkdir -p ${package_dir}/final_dist
if command -v dnf >/dev/null 2>&1 ; then
    export POLICY="manylinux_2_28"
    export REAL_ARCH=$(arch)
    export AUDITWHEEL_POLICY=${POLICY}
    export AUDITWHEEL_ARCH=${REAL_ARCH}
    export AUDITWHEEL_PLAT=${POLICY}_${REAL_ARCH}
    manylinux="manylinux_2_28"
fi
python -m auditwheel repair -w ${package_dir}/final_dist ${package_dir}/dist/*

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_${manylinux}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
