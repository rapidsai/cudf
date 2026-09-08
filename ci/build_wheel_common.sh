#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

source rapids-configure-sccache
source rapids-datetime-string
source rapids-init-pip

export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

RAPIDS_VERSION_SUFFIX=".post${RAPIDS_DATETIME_STRING}" \
  rapids-generate-version | tee ./VERSION > ./python/cudf/cudf/VERSION

build_package_wheel() {
  local package_key=$1
  local package_name=$2
  local package_dir=$3
  shift 3
  local build_log=

  if [[ "${1:-}" == "--log" ]]; then
    build_log=$2
    shift 2
  fi

  export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}-${RAPIDS_CONDA_ARCH}-cuda${RAPIDS_CUDA_VERSION%%.*}-wheel-preprocessor-cache"
  export RAPIDS_WHEEL_BLD_OUTPUT_DIR="${PWD}/wheel-output/${package_key}"
  mkdir -p "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  if [[ -n "${build_log}" ]]; then
    ./ci/build_wheel.sh "${package_name}" "${package_dir}" "$@" 2>&1 | tee "${build_log}"
  else
    ./ci/build_wheel.sh "${package_name}" "${package_dir}" "$@"
  fi
}

record_wheel_artifact() {
  local package_key=$1
  local package_name=$2
  {
    echo "${package_key}_artifact_name=${package_name}"
    echo "${package_key}_output_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  } >> "${GITHUB_OUTPUT}"
}

finalize_package_wheel() {
  local package_key=$1
  local package_dir=$2
  local max_wheel_size=$3
  local artifact_name=$4

  ./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" "${max_wheel_size}"
  record_wheel_artifact "${package_key}" "${artifact_name}"
}
