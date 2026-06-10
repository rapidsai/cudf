#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="cudf-streaming"
package_dir="python/cudf_streaming"
dependency_file_key_suffix="cudf_streaming"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcudf_streaming wheel from this current build,
# then ensures 'cudf_streaming' wheel builds always use the 'libcudf_streaming' just built in the same CI run.
LIBCUDF_STREAMING_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcudf_streaming_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibcudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
echo "libcudf-streaming-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_*.whl)" >> "${PIP_CONSTRAINT}"
echo "pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_*.whl)" >> "${PIP_CONSTRAINT}"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${dependency_file_key_suffix}" \
  --file-key "py_rapids_build_${dependency_file_key_suffix}" \
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

# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

./ci/build_wheel.sh "${package_name}" "${package_dir}" --stable

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    --exclude libcudf.so \
    --exclude libcudf_streaming.so \
    --exclude librapidsmpf.so \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude libucxx.so \
    --exclude libucp.so.0 \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-package-name wheel_python cudf_streaming --stable --cuda)"
export RAPIDS_PACKAGE_NAME
