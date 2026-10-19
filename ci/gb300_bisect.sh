#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDA_MAJOR="${RAPIDS_PY_CUDA_SUFFIX#cu}"
RAPIDS_VERSION="$(rapids-version-major-minor)"

rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

# cudf wheels from the main build of 816f682244, which segfaults on GB300 with current deps.
CUDF_SHA=816f682244c080b438792eec6bc8eda6c3b0583a
CUDF_RUN_ID=39100160924

rmm_pins() {
  echo "rmm-${RAPIDS_PY_CUDA_SUFFIX}==$1 librmm-${RAPIDS_PY_CUDA_SUFFIX}==$1"
}
kvikio_pins() {
  echo "kvikio-${RAPIDS_PY_CUDA_SUFFIX}==$1 libkvikio-${RAPIDS_PY_CUDA_SUFFIX}==$1"
}

CONFIGS=(
  "latest|"
  "rmm-a26+kvikio-a29|$(rmm_pins ${RAPIDS_VERSION}.0a26.post261015050821) $(kvikio_pins ${RAPIDS_VERSION}.0a29.post261015050825)"
  "rmm-a26|$(rmm_pins ${RAPIDS_VERSION}.0a26.post261015050821)"
  "kvikio-a29|$(kvikio_pins ${RAPIDS_VERSION}.0a29.post261015050825)"
  "cuda-core-1.2.1|cuda-core==1.2.1"
  "rmm-a27|$(rmm_pins ${RAPIDS_VERSION}.0a27.post261015192231)"
  "rmm-a28|$(rmm_pins ${RAPIDS_VERSION}.0a28.post261016050759)"
  "rmm-a29|$(rmm_pins ${RAPIDS_VERSION}.0a29.post261016155223)"
  "rmm-a30|$(rmm_pins ${RAPIDS_VERSION}.0a30.post261017050732)"
  "rmm-a31|$(rmm_pins ${RAPIDS_VERSION}.0a31.post261019050850)"
)

WORK="$(mktemp -d)"
mkdir -p "${WORK}/wheels" "${WORK}/src"
gh run download "${CUDF_RUN_ID}" --repo NVIDIA/cudf \
    --name "cudf_wheel_cpp_libcudf_aarch64_cu${CUDA_MAJOR}" \
    --name "cudf_wheel_python_pylibcudf_aarch64_abi3_cu${CUDA_MAJOR}" \
    --dir "${WORK}/wheels"
git archive "${CUDF_SHA}" python/pylibcudf/tests | tar -x -C "${WORK}/src"

SUMMARY=()
for config in "${CONFIGS[@]}"; do
  label="${config%%|*}"
  read -r -a pins <<< "${config#*|}"
  env_dir="${WORK}/env-${label}"

  rapids-logger "Config ${label}: ${pins[*]:-}"

  python -m venv "${env_dir}"
  # shellcheck disable=SC1091
  . "${env_dir}/bin/activate"
  if ! rapids-pip-retry install \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(find "${WORK}/wheels" -name "libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")" \
      "$(find "${WORK}/wheels" -name "pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")[test]" \
      "${pins[@]}"; then
    SUMMARY+=("${label} INSTALL_FAILED")
    deactivate
    continue
  fi
  python -m pip list 2>/dev/null | grep -i -E "rmm|kvikio|cuda-core|cudf"

  for attempt in 1 2; do
    pushd "${WORK}/src/python/pylibcudf/tests" > /dev/null
    set +e
    timeout 15m python -m pytest \
      --cache-clear \
      --numprocesses=8 \
      --dist=worksteal \
      -q \
      . 2>&1 | tee "${env_dir}/pytest-${attempt}.log"
    status=${PIPESTATUS[0]}
    set -e
    popd > /dev/null
    crashed=$(grep -o "crashed while running '[^']*'" "${env_dir}/pytest-${attempt}.log" | sort -u | tr '\n' ' ' || true)
    SUMMARY+=("${label} attempt=${attempt} exit=${status} ${crashed}")
  done

  deactivate
  rm -rf "${env_dir}"
done

rapids-logger "GB300 dependency bisect summary"
printf '%s\n' "${SUMMARY[@]}"
