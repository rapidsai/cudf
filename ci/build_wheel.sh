#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

# Build all non-noarch wheels in one local dependency chain.

# libcudf
SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON" \
  build_package_wheel libcudf libcudf python/libcudf

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
python -m auditwheel repair \
  --exclude libkvikio.so \
  --exclude libnvcomp.so.5 \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude "libnvrtc.so.${RAPIDS_CUDA_MAJOR}" \
  --exclude "libnvJitLink.so.${RAPIDS_CUDA_MAJOR}" \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/libcudf/dist/*

WHEEL_EXPORT_DIR="$(mktemp -d)"
unzip -d "${WHEEL_EXPORT_DIR}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*
LIBCUDF_LIBRARY="$(find "${WHEEL_EXPORT_DIR}" -type f -name libcudf.so)"
./ci/check_symbols.sh "${LIBCUDF_LIBRARY}"

if [[ "${RAPIDS_CUDA_MAJOR}" == "12" ]]; then
  libcudf_max_wheel_size=700M
else
  libcudf_max_wheel_size=350M
fi
finalize_package_wheel \
  libcudf \
  python/libcudf \
  "${libcudf_max_wheel_size}" \
  "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "${RAPIDS_CUDA_VERSION}")"

# libcudf-streaming uses the libcudf wheel built above.
export RAPIDS_LIBCUDF_WHEELHOUSE="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
LIBCUDF_WHEELHOUSE="${RAPIDS_LIBCUDF_WHEELHOUSE}"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"

build_package_wheel libcudf_streaming libcudf_streaming python/libcudf_streaming

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude librapidsmpf.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude libucxx.so \
  --exclude libucp.so.0 \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/libcudf_streaming/dist/*

finalize_package_wheel \
  libcudf_streaming \
  python/libcudf_streaming \
  100M \
  "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "${RAPIDS_CUDA_VERSION}")"

export RAPIDS_LIBCUDF_STREAMING_WHEELHOUSE="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

BASE_PIP_CONSTRAINT="$(mktemp)"
cp "${PIP_CONSTRAINT}" "${BASE_PIP_CONSTRAINT}"
trap 'rm -f "${BASE_PIP_CONSTRAINT}"' EXIT

# All wheels in this stage use the stable Python ABI.
export RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"

# pylibcudf
cp "${BASE_PIP_CONSTRAINT}" "${PIP_CONSTRAINT}"
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"
build_package_wheel \
  pylibcudf \
  pylibcudf \
  python/pylibcudf \
  --log pylibcudf-wheel-build-output.log \
  --stable
check_cython_performance_hints pylibcudf pylibcudf-wheel-build-output.log

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/pylibcudf/dist/*

finalize_package_wheel \
  pylibcudf \
  python/pylibcudf \
  20M \
  "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"

# cudf
PYLIBCUDF_WHEELHOUSE="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
cp "${BASE_PIP_CONSTRAINT}" "${PIP_CONSTRAINT}"
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"
echo "pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_*.whl)" >> "${PIP_CONSTRAINT}"
build_package_wheel cudf cudf python/cudf --stable

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/cudf/dist/*

finalize_package_wheel \
  cudf \
  python/cudf \
  15M \
  "$(rapids-artifact-name wheel_python cudf cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"

# cudf-streaming
cp "${BASE_PIP_CONSTRAINT}" "${PIP_CONSTRAINT}"
LIBCUDF_STREAMING_WHEELHOUSE="${RAPIDS_LIBCUDF_STREAMING_WHEELHOUSE}"
echo "libcudf-streaming-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_STREAMING_WHEELHOUSE}"/libcudf_streaming_*.whl)" >> "${PIP_CONSTRAINT}"
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"
echo "pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_*.whl)" >> "${PIP_CONSTRAINT}"

build_package_wheel \
  cudf_streaming \
  cudf-streaming \
  python/cudf_streaming \
  --log cudf-streaming-wheel-build-output.log \
  --stable
check_cython_performance_hints cudf-streaming cudf-streaming-wheel-build-output.log

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude libcudf_streaming.so \
  --exclude librapidsmpf.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude libucxx.so \
  --exclude libucp.so.0 \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/cudf_streaming/dist/*

finalize_package_wheel \
  cudf_streaming \
  python/cudf_streaming \
  75M \
  "$(rapids-artifact-name wheel_python cudf-streaming cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"
