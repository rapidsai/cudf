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

CONFIGS=(
  "baseline|"
  "per-thread-stream|CUDF_PER_THREAD_STREAM=1"
  "nvcomp-pinned-pool-0|NVCOMP_PINNED_POOL_SIZE=0"
)

SUMMARY=()
pushd python/pylibcudf/tests > /dev/null
for config in "${CONFIGS[@]}"; do
  label="${config%%|*}"
  read -r -a envs <<< "${config#*|}"
  for attempt in 1 2 3 4; do
    set +e
    env "${envs[@]}" timeout 15m python -m pytest -p no:cacheprovider --numprocesses=8 --dist=worksteal -q . \
      > "${WORK}/${label}-${attempt}.log" 2>&1
    status=$?
    set -e
    crash=$(grep -o "crashed while running '[^']*'" "${WORK}/${label}-${attempt}.log" | sort -u | tr '\n' ' ' || true)
    SUMMARY+=("${label} attempt=${attempt} exit=${status} ${crash}")
  done
done

rapids-logger "nvCOMP trace logging"
for attempt in 1 2 3 4; do
  rm -f "${WORK}/nvcomp.log"
  set +e
  NVCOMP_LOG_LEVEL=5 NVCOMP_LOG_FILE="${WORK}/nvcomp.log" timeout 15m python -m pytest -p no:cacheprovider \
    --numprocesses=8 --dist=worksteal -q . > "${WORK}/nvcomp-trace-${attempt}.log" 2>&1
  status=$?
  set -e
  crash=$(grep -o "crashed while running '[^']*'" "${WORK}/nvcomp-trace-${attempt}.log" | sort -u | tr '\n' ' ' || true)
  SUMMARY+=("nvcomp-trace attempt=${attempt} exit=${status} ${crash}")
  rapids-logger "nvCOMP log attempt ${attempt}: $(wc -l < "${WORK}/nvcomp.log" 2>/dev/null || echo 0) lines"
  if [[ -f "${WORK}/nvcomp.log" ]]; then
    cut -d']' -f2-3 "${WORK}/nvcomp.log" | sort | uniq -c | sort -rn | head -n 20 || true
    grep -i -E "warn|error|fail|fall" "${WORK}/nvcomp.log" | sort | uniq -c | sort -rn | head -n 40 || true
  fi
  if [[ -n "${crash}" ]]; then
    rapids-logger "nvCOMP log tail around the crash"
    tail -n 80 "${WORK}/nvcomp.log" || true
    break
  fi
done
popd > /dev/null

rapids-logger "GB300 DE workaround probe summary"
printf '%s\n' "${SUMMARY[@]}"
