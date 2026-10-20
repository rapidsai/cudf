#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# The fix only changes Python tests, so use the main build of this branch's base commit
# (4473c24036) instead of building wheels in this PR.
WHEELHOUSE="$(mktemp -d)"
gh run download 39273725253 --repo NVIDIA/cudf \
    --name "cudf_wheel_cpp_libcudf_aarch64_${RAPIDS_PY_CUDA_SUFFIX}" \
    --name "cudf_wheel_python_pylibcudf_aarch64_abi3_${RAPIDS_PY_CUDA_SUFFIX}" \
    --dir "${WHEELHOUSE}"
LIBCUDF_WHEELHOUSE="${WHEELHOUSE}"
PYLIBCUDF_WHEELHOUSE="${WHEELHOUSE}"

rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

python -m venv pylibcudf-env
# shellcheck disable=SC1091
. pylibcudf-env/bin/activate
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${PYLIBCUDF_WHEELHOUSE}"/pylibcudf_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

WORK="$(mktemp -d)"
SUMMARY=()
FIX_CRASHES=0

run_suite() {
  local label=$1
  shift
  set +e
  env "$@" timeout 15m python -m pytest -p no:cacheprovider --numprocesses=8 --dist=worksteal -q . \
    > "${WORK}/${label}.log" 2>&1
  local status=$?
  set -e
  tail -n 1 "${WORK}/${label}.log"
  local crash
  crash=$(grep -o "crashed while running '[^']*'" "${WORK}/${label}.log" | sort -u | tr '\n' ' ' || true)
  SUMMARY+=("${label} exit=${status} ${crash}")
  [[ -z "${crash}" && ${status} -eq 0 ]]
}

pushd python/pylibcudf/tests > /dev/null
for attempt in 1 2 3 4 5 6; do
  run_suite "fix-${attempt}" || FIX_CRASHES=$((FIX_CRASHES + 1))
done
popd > /dev/null

# Standalone reproduction of the root cause: each process sets up the given memory pool scenario,
# then reads a Snappy parquet file (nvCOMP hardware decompression). 8 concurrent processes x 3.
REPRO="$(realpath ci/gb300_repro.py)"
for mode in none rmm-async-destroy-pending rmm-async-destroy-synced \
  raw-plain-max-threshold-written-pending raw-plain-max-threshold-written-synced \
  raw-plain-max-threshold-written-pending-hw-off; do
  crashed=0
  for round in 1 2 3; do
    pids=()
    for i in $(seq 8); do
      timeout 5m python -X faulthandler "${REPRO}" "${mode}" > "${WORK}/${mode}-${round}-${i}.log" 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
    for i in $(seq 8); do
      if ! grep -q "^OK ${mode}$" "${WORK}/${mode}-${round}-${i}.log"; then
        crashed=$((crashed + 1))
        if [[ ${crashed} -eq 1 ]]; then
          rapids-logger "repro ${mode}: first failing process output"
          grep -v -E "^\s*$" "${WORK}/${mode}-${round}-${i}.log" | head -n 25 || true
        fi
      fi
    done
  done
  SUMMARY+=("repro ${mode} failed_processes=${crashed}/24")
done

rapids-logger "GB300 fix validation summary"
printf '%s\n' "${SUMMARY[@]}"
echo "Runs with crashes or test failures: ${FIX_CRASHES}/6"
[[ ${FIX_CRASHES} -eq 0 ]]
