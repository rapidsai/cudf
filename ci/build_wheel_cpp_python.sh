#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_common.sh
source ./ci/build_wheel_common.sh

build_cpp_wheels() {
  # libcudf
  SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON" \
    build_package_wheel libcudf libcudf python/libcudf

  local rapids_cuda_major="${RAPIDS_CUDA_VERSION%%.*}"
  python -m auditwheel repair \
    --exclude libkvikio.so \
    --exclude libnvcomp.so.5 \
    --exclude librapids_logger.so \
    --exclude librmm.so \
    --exclude "libnvrtc.so.${rapids_cuda_major}" \
    --exclude "libnvJitLink.so.${rapids_cuda_major}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    python/libcudf/dist/*

  local wheel_export_dir
  wheel_export_dir="$(mktemp -d)"
  unzip -d "${wheel_export_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*
  local libcudf_library
  libcudf_library="$(find "${wheel_export_dir}" -type f -name libcudf.so)"
  ./ci/check_symbols.sh "${libcudf_library}"

  local libcudf_max_wheel_size
  if [[ "${rapids_cuda_major}" == "12" ]]; then
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
  local libcudf_wheelhouse="${RAPIDS_LIBCUDF_WHEELHOUSE}"

  local rapids_py_cuda_suffix
  rapids_py_cuda_suffix="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
  echo "libcudf-${rapids_py_cuda_suffix} @ file://$(echo "${libcudf_wheelhouse}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"

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
}

check_cython_performance_hints() {
  local package_name=$1
  local build_log=$2

  rapids-logger "Checking for Cython performance warnings"
  if grep -Fq "performance hint:" "${build_log}"; then
    echo "Cython performance hints found in ${package_name} build:"
    grep -F "performance hint:" "${build_log}"
    exit 1
  fi
}

build_python_wheels() {
  BASE_PIP_CONSTRAINT="$(mktemp)"
  cp "${PIP_CONSTRAINT}" "${BASE_PIP_CONSTRAINT}"
  trap 'rm -f "${BASE_PIP_CONSTRAINT}"' EXIT

  local rapids_py_cuda_suffix
  rapids_py_cuda_suffix="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
  # All wheels in this stage use the stable Python ABI.
  export RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"

  # pylibcudf
  cp "${BASE_PIP_CONSTRAINT}" "${PIP_CONSTRAINT}"
  local libcudf_wheelhouse="${RAPIDS_LIBCUDF_WHEELHOUSE:-$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "${RAPIDS_CUDA_VERSION}")")}"
  echo "libcudf-${rapids_py_cuda_suffix} @ file://$(echo "${libcudf_wheelhouse}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"
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
  local pylibcudf_wheelhouse="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  cp "${BASE_PIP_CONSTRAINT}" "${PIP_CONSTRAINT}"
  echo "libcudf-${rapids_py_cuda_suffix} @ file://$(echo "${libcudf_wheelhouse}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"
  echo "pylibcudf-${rapids_py_cuda_suffix} @ file://$(echo "${pylibcudf_wheelhouse}"/pylibcudf_*.whl)" >> "${PIP_CONSTRAINT}"
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
  local libcudf_streaming_wheelhouse="${RAPIDS_LIBCUDF_STREAMING_WHEELHOUSE:-$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "${RAPIDS_CUDA_VERSION}")")}"
  echo "libcudf-streaming-${rapids_py_cuda_suffix} @ file://$(echo "${libcudf_streaming_wheelhouse}"/libcudf_streaming_*.whl)" >> "${PIP_CONSTRAINT}"
  echo "libcudf-${rapids_py_cuda_suffix} @ file://$(echo "${libcudf_wheelhouse}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"
  echo "pylibcudf-${rapids_py_cuda_suffix} @ file://$(echo "${pylibcudf_wheelhouse}"/pylibcudf_*.whl)" >> "${PIP_CONSTRAINT}"

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
}

case "${1:-cpp-python}" in
  cpp)
    build_cpp_wheels
    ;;
  python)
    build_python_wheels
    ;;
  cpp-python)
    build_cpp_wheels
    build_python_wheels
    ;;
  *)
    echo "Usage: $0 [cpp|python|cpp-python]" >&2
    exit 2
    ;;
esac
