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

CONFIGS=(
  "baseline|"
  "hw-decompression-off|LIBCUDF_HW_DECOMPRESSION=OFF"
  "hw-decompression-on|LIBCUDF_HW_DECOMPRESSION=ON"
  "nvcomp-off|LIBCUDF_NVCOMP_POLICY=OFF"
  "kvikio-1-thread|KVIKIO_NTHREADS=1"
  "launch-blocking|CUDA_LAUNCH_BLOCKING=1"
)
ATTEMPTS=4

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

pushd "${WORK}/src/python/pylibcudf/tests" > /dev/null

SUMMARY=()
for config in "${CONFIGS[@]}"; do
  label="${config%%|*}"
  read -r -a envs <<< "${config#*|}"
  rapids-logger "Config ${label}: ${envs[*]:-}"
  for attempt in $(seq "${ATTEMPTS}"); do
    set +e
    env "${envs[@]}" timeout 15m python -m pytest \
      --cache-clear \
      --numprocesses=8 \
      --dist=worksteal \
      -q \
      . > "${WORK}/${label}-${attempt}.log" 2>&1
    status=$?
    set -e
    tail -n 3 "${WORK}/${label}-${attempt}.log"
    crashed=$(grep -o "crashed while running '[^']*'" "${WORK}/${label}-${attempt}.log" | sort -u | tr '\n' ' ' || true)
    SUMMARY+=("${label} attempt=${attempt} exit=${status} ${crashed}")
  done
done

rapids-logger "Single-process gdb backtraces of the segfault"
apt-get update -qq && apt-get install -y -qq --no-install-recommends gdb > /dev/null
for attempt in 1 2 3; do
  set +e
  timeout 10m gdb -q -batch \
    -ex "handle SIGSEGV stop print nopass" \
    -ex "set pagination off" \
    -ex run \
    -ex "info threads" \
    -ex "thread apply all bt 25" \
    --args python -m pytest --cache-clear -p no:xdist -q io/test_parquet.py \
    > "${WORK}/gdb-${attempt}.log" 2>&1
  set -e
  if grep -q "SIGSEGV" "${WORK}/gdb-${attempt}.log"; then
    grep -v -E "^\[(New|Thread) .* (exited|Thread)" "${WORK}/gdb-${attempt}.log" | sed -n '/SIGSEGV/,$p'
    SUMMARY+=("gdb attempt=${attempt} SIGSEGV")
    break
  fi
  SUMMARY+=("gdb attempt=${attempt} no crash")
done

popd > /dev/null

rapids-logger "GB300 diagnostics summary"
printf '%s\n' "${SUMMARY[@]}"
