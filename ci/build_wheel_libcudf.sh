#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

package_name="libcudf"
package_dir="python/libcudf"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

# TODO(nvcomp): when `nvcomp` supports Python 3.13 and we de-vendor `nvcomp` from `kvikio`
# this should be switched back to using the nvcomp runtime wheel
# https://github.com/rapidsai/build-planning/issues/171
# export SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON"
export SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_FROM_LIBKVIKIO_WHEEL=ON"
./ci/build_wheel.sh "${package_name}" "${package_dir}"

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    --exclude libnvcomp.so.4 \
    --exclude libkvikio.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

WHEEL_EXPORT_DIR="$(mktemp -d)"
unzip -d "${WHEEL_EXPORT_DIR}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/*"
LIBCUDF_LIBRARY=$(find "${WHEEL_EXPORT_DIR}" -type f -name 'libcudf.so')
./ci/check_symbols.sh "${LIBCUDF_LIBRARY}"

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
