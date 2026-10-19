#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDA_MAJOR="${RAPIDS_PY_CUDA_SUFFIX#cu}"

rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

# cudf wheels from the main build of 816f682244, which segfaults on GB300.
CUDF_SHA=816f682244c080b438792eec6bc8eda6c3b0583a
CUDF_RUN_ID=39100160924

WORK="$(mktemp -d)"
mkdir -p "${WORK}/wheels" "${WORK}/src"
gh run download "${CUDF_RUN_ID}" --repo NVIDIA/cudf \
    --name "cudf_wheel_cpp_libcudf_aarch64_cu${CUDA_MAJOR}" \
    --name "cudf_wheel_python_pylibcudf_aarch64_abi3_cu${CUDA_MAJOR}" \
    --dir "${WORK}/wheels"
git archive "${CUDF_SHA}" python/pylibcudf/tests | tar -x -C "${WORK}/src"

python -m venv "${WORK}/env"
# shellcheck disable=SC1091
. "${WORK}/env/bin/activate"
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(find "${WORK}/wheels" -name "libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")" \
    "$(find "${WORK}/wheels" -name "pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")[test]"
python -m pip list 2>/dev/null | grep -i -E "rmm|kvikio|nvcomp|cuda|cudf"

REPRO="$(realpath ci/gb300_repro.py)"
SUMMARY=()

run_repro() {
  local label=$1 mode=$2
  shift 2
  for attempt in 1 2 3 4 5; do
    set +e
    env "$@" timeout 5m python -X faulthandler "${REPRO}" "${mode}" 20 > "${WORK}/${label}-${attempt}.log" 2>&1
    local status=$?
    set -e
    SUMMARY+=("${label} attempt=${attempt} exit=${status}")
    if [[ ${status} -ne 0 ]]; then
      rapids-logger "${label} attempt ${attempt} failed with ${status}"
      grep -v -E "^\s*$" "${WORK}/${label}-${attempt}.log" | head -n 60 || true
    fi
  done
}

for mode in path bytes; do
  run_repro "${mode}-default" "${mode}"
  run_repro "${mode}-launch-blocking" "${mode}" CUDA_LAUNCH_BLOCKING=1
  run_repro "${mode}-hw-off" "${mode}" LIBCUDF_HW_DECOMPRESSION=OFF
  run_repro "${mode}-hw-off-launch-blocking" "${mode}" LIBCUDF_HW_DECOMPRESSION=OFF CUDA_LAUNCH_BLOCKING=1
done

rapids-logger "8 concurrent processes, path mode"
for round in 1 2 3; do
  pids=()
  for i in $(seq 8); do
    python -X faulthandler "${REPRO}" path 50 > "${WORK}/concurrent-${round}-${i}.log" 2>&1 &
    pids+=($!)
  done
  failures=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failures=$((failures + 1))
      grep -v -E "^\s*$" "${WORK}/concurrent-${round}-$((i + 1)).log" | head -n 40 || true
    fi
  done
  SUMMARY+=("concurrent round=${round} failed_processes=${failures}/8")
done

rapids-logger "gdb backtrace, path mode with CUDA_LAUNCH_BLOCKING=1"
apt-get update -qq && apt-get install -y -qq --no-install-recommends gdb > /dev/null
set +e
CUDA_LAUNCH_BLOCKING=1 timeout 10m gdb -q -batch \
  -ex "handle SIGSEGV stop print nopass" \
  -ex "set pagination off" \
  -ex run \
  -ex "info sharedlibrary nvcomp" \
  -ex "info threads" \
  -ex "thread apply all bt 30" \
  --args python "${REPRO}" path 20 > "${WORK}/gdb.log" 2>&1
set -e
grep -v -E "^\[(New Thread|Thread .* exited)" "${WORK}/gdb.log" | sed -n '/SIGSEGV/,$p' | head -n 400 || true

rapids-logger "GB300 repro summary"
printf '%s\n' "${SUMMARY[@]}"
