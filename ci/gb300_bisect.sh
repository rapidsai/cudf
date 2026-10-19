#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDA_MAJOR="${RAPIDS_PY_CUDA_SUFFIX#cu}"

rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

# main builds immediately before (nvCOMP 5.3) and after (nvCOMP 6.0) the nvCOMP 6 upgrade in #24580.
BUILDS=(
  "nvcomp-5.3-dd59a8999d|38657747679"
  "nvcomp-6.0-7d90ae098c|38665680183"
  "nvcomp-6.0-ed48829288|38674723219"
)

WORK="$(mktemp -d)"
SUMMARY=()
for build in "${BUILDS[@]}"; do
  label="${build%%|*}"
  run_id="${build##*|}"
  dir="${WORK}/${label}"
  mkdir -p "${dir}/wheels"
  gh run download "${run_id}" --repo NVIDIA/cudf \
      --name "cudf_wheel_cpp_libcudf_aarch64_cu${CUDA_MAJOR}" \
      --name "cudf_wheel_python_pylibcudf_aarch64_abi3_cu${CUDA_MAJOR}" \
      --dir "${dir}/wheels"

  python -m venv "${dir}/env"
  # shellcheck disable=SC1091
  . "${dir}/env/bin/activate"
  rapids-pip-retry install \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(find "${dir}/wheels" -name "libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")" \
      "$(find "${dir}/wheels" -name "pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")[test]"
  python -m pip list 2>/dev/null | grep -i -E "nvcomp|cudf"

  pushd python/pylibcudf/tests > /dev/null
  for attempt in 1 2 3 4; do
    set +e
    timeout 15m python -m pytest -p no:cacheprovider --numprocesses=8 --dist=worksteal -q . \
      > "${dir}/pytest-${attempt}.log" 2>&1
    status=$?
    set -e
    crash=$(grep -o "crashed while running '[^']*'" "${dir}/pytest-${attempt}.log" | sort -u | tr '\n' ' ' || true)
    SUMMARY+=("${label} attempt=${attempt} exit=${status} ${crash}")
  done
  popd > /dev/null

  deactivate
  rm -rf "${dir}"
done

rapids-logger "GB300 nvCOMP version comparison summary"
printf '%s\n' "${SUMMARY[@]}"
