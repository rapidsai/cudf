#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDA_MAJOR="${RAPIDS_PY_CUDA_SUFFIX#cu}"

rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

# cudf wheels from the main build of 816f682244 (nvCOMP 6.0), which segfaults on GB300.
CUDF_RUN_ID=39100160924

WORK="$(mktemp -d)"
mkdir -p "${WORK}/wheels"
gh run download "${CUDF_RUN_ID}" --repo NVIDIA/cudf \
    --name "cudf_wheel_cpp_libcudf_aarch64_cu${CUDA_MAJOR}" \
    --name "cudf_wheel_python_pylibcudf_aarch64_abi3_cu${CUDA_MAJOR}" \
    --dir "${WORK}/wheels"

python -m venv "${WORK}/env"
# shellcheck disable=SC1091
. "${WORK}/env/bin/activate"
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(find "${WORK}/wheels" -name "libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")" \
    "$(find "${WORK}/wheels" -name "pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")[test]"

REPRO="$(realpath ci/gb300_repro.py)"
SUMMARY=()
for config in rmm-async-destroy rmm-async-sync-before-destroy rmm-async-noalloc-destroy \
  raw-hw-pool-destroy-pending raw-plain-pool-destroy-pending rmm-async-destroy:LIBCUDF_HW_DECOMPRESSION=OFF; do
  mode="${config%%:*}"
  envs=()
  if [[ "${config}" == *:* ]]; then envs=("${config#*:}"); fi
  label="${config//[:=]/-}"
  crashed=0
  for round in 1 2 3; do
    pids=()
    for i in $(seq 8); do
      env "${envs[@]}" timeout 5m python -X faulthandler "${REPRO}" "${mode}" > "${WORK}/${label}-${round}-${i}.log" 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
    for i in $(seq 8); do
      if ! grep -q "^OK ${mode}$" "${WORK}/${label}-${round}-${i}.log"; then
        crashed=$((crashed + 1))
        if [[ ${crashed} -eq 1 ]]; then
          rapids-logger "${label}: first failing process output"
          grep -v -E "^\s*$" "${WORK}/${label}-${round}-${i}.log" | head -n 30 || true
        fi
      fi
    done
  done
  SUMMARY+=("${label} failed_processes=${crashed}/24")
done

rapids-logger "GB300 HW-decompress pool repro summary"
printf '%s\n' "${SUMMARY[@]}"
