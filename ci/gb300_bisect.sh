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

NODEIDS="$(realpath ci/gb300_gw6_nodeids.txt)"
mkdir -p "${WORK}/src"
git archive 816f682244c080b438792eec6bc8eda6c3b0583a python/pylibcudf/tests | tar -x -C "${WORK}/src"

replay() {
  python -X faulthandler -c "
import sys

import pytest

ids = open(sys.argv[1]).read().splitlines()
sys.exit(pytest.main(['-p', 'no:xdist', '-p', 'no:cacheprovider', '-q', *ids]))
" "${NODEIDS}"
}

SUMMARY=()
pushd "${WORK}/src/python/pylibcudf/tests" > /dev/null
for attempt in 1 2 3; do
  set +e
  replay > "${WORK}/serial-${attempt}.log" 2>&1
  status=$?
  set -e
  tail -n 1 "${WORK}/serial-${attempt}.log"
  crashed=""
  if grep -q "Fatal Python error" "${WORK}/serial-${attempt}.log"; then crashed="SEGFAULT"; fi
  SUMMARY+=("serial-replay attempt=${attempt} exit=${status} ${crashed}")
done

for round in 1 2; do
  pids=()
  for i in $(seq 8); do
    replay > "${WORK}/concurrent-${round}-${i}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
  crashed=0
  for i in $(seq 8); do
    if grep -q "Fatal Python error" "${WORK}/concurrent-${round}-${i}.log"; then
      crashed=$((crashed + 1))
      if [[ ${crashed} -eq 1 ]]; then
        grep -B 3 "Fatal Python error" "${WORK}/concurrent-${round}-${i}.log" | head -n 5 || true
        sed -n '/Current thread/,/Current thread.s C stack/p' "${WORK}/concurrent-${round}-${i}.log" | head -n 8 || true
      fi
    fi
  done
  SUMMARY+=("concurrent-replay round=${round} crashed_processes=${crashed}/8")
done
popd > /dev/null

rapids-logger "GB300 exact gw6 replay summary"
printf '%s\n' "${SUMMARY[@]}"
