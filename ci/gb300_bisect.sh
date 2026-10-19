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

pushd "${WORK}/src/python/pylibcudf/tests" > /dev/null
SUMMARY=()

rapids-logger "Serial collection check"
python -m pytest -p no:xdist -p no:cacheprovider --collect-only -q . 2>&1 | tail -n 30 || true

run_xdist_verbose() {
  local label=$1
  shift
  for attempt in 1 2 3; do
    set +e
    env "$@" timeout 15m python -m pytest -p no:cacheprovider --numprocesses=8 --dist=worksteal -v . \
      > "${WORK}/${label}-${attempt}.log" 2>&1
    local status=$?
    set -e
    local crash
    crash=$(grep -o "worker '[^']*' crashed while running '[^']*'" "${WORK}/${label}-${attempt}.log" | head -n 1 || true)
    SUMMARY+=("${label} attempt=${attempt} exit=${status} ${crash}")
    if [[ -n "${crash}" ]]; then
      local worker
      worker=$(sed -E "s/worker '([^']*)'.*/\1/" <<< "${crash}")
      rapids-logger "${label} attempt ${attempt}: tests completed by ${worker} before it crashed"
      grep -E "^\[${worker}\]" "${WORK}/${label}-${attempt}.log" || true
      rapids-logger "${label} attempt ${attempt}: crash stack"
      sed -n '/Fatal Python error/,/Extension modules/p' "${WORK}/${label}-${attempt}.log" | head -n 80 || true
      return
    fi
  done
}

run_xdist_verbose launch-blocking CUDA_LAUNCH_BLOCKING=1
run_xdist_verbose launch-blocking-hw-off CUDA_LAUNCH_BLOCKING=1 LIBCUDF_HW_DECOMPRESSION=OFF
run_xdist_verbose default

popd > /dev/null

rapids-logger "GB300 worker-sequence summary"
printf '%s\n' "${SUMMARY[@]}"
