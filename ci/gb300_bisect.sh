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

run_serial() {
  local label=$1 target=$2
  shift 2
  set +e
  env "$@" timeout 20m python -m pytest -p no:xdist -p no:cacheprovider -v "${target}" > "${WORK}/${label}.log" 2>&1
  local status=$?
  set -e
  local last
  last=$(grep -E "(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)" "${WORK}/${label}.log" | tail -n 1 || true)
  SUMMARY+=("${label} exit=${status} last_completed=${last}")
  rapids-logger "${label} exit=${status}"
  if grep -q "Fatal Python error" "${WORK}/${label}.log"; then
    grep -B 15 "Fatal Python error" "${WORK}/${label}.log" | grep -E "::" || true
    sed -n '/Fatal Python error/,/Extension modules/p' "${WORK}/${label}.log" | head -n 80 || true
  fi
}

run_serial suite-launch-blocking . CUDA_LAUNCH_BLOCKING=1
run_serial suite-launch-blocking-hw-off . CUDA_LAUNCH_BLOCKING=1 LIBCUDF_HW_DECOMPRESSION=OFF
run_serial io-launch-blocking io CUDA_LAUNCH_BLOCKING=1
run_serial test-types-launch-blocking io/test_types.py CUDA_LAUNCH_BLOCKING=1
run_serial suite-default .
run_serial suite-hw-on . LIBCUDF_HW_DECOMPRESSION=ON

popd > /dev/null

rapids-logger "GB300 serial summary"
printf '%s\n' "${SUMMARY[@]}"
