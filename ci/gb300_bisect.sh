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
VARIANTS=(
  "read-only none"
  "orc none"
  "orc persistent"
  "orc temporary"
  "parquet none"
  "parquet persistent"
  "parquet temporary"
)

SUMMARY=()
for variant in "${VARIANTS[@]}"; do
  read -r writer stream_kind <<< "${variant}"
  label="${writer}-${stream_kind}"
  crashes=0
  for attempt in $(seq 10); do
    set +e
    timeout 5m python -X faulthandler "${REPRO}" "${writer}" "${stream_kind}" 20 > "${WORK}/${label}-${attempt}.log" 2>&1
    status=$?
    set -e
    if [[ ${status} -ne 0 ]]; then
      crashes=$((crashes + 1))
      if [[ ${crashes} -eq 1 ]]; then
        rapids-logger "${label} attempt ${attempt} failed with ${status}"
        grep -v -E "^\s*$" "${WORK}/${label}-${attempt}.log" | head -n 50 || true
      fi
    fi
  done
  SUMMARY+=("${label} failed_processes=${crashes}/10")
done

rapids-logger "8 concurrent processes: orc writer with temporary streams"
pids=()
for i in $(seq 8); do
  python -X faulthandler "${REPRO}" orc temporary 50 > "${WORK}/concurrent-${i}.log" 2>&1 &
  pids+=($!)
done
failures=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failures=$((failures + 1))
    grep -v -E "^\s*$" "${WORK}/concurrent-$((i + 1)).log" | head -n 30 || true
  fi
done
SUMMARY+=("concurrent orc-temporary failed_processes=${failures}/8")

rapids-logger "GB300 compress-then-DE repro summary"
printf '%s\n' "${SUMMARY[@]}"
