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

SUMMARY=()
pushd python/pylibcudf/tests > /dev/null
rapids-logger "nvCOMP trace logging"
for attempt in 1 2 3 4 5 6; do
  rm -f "${WORK}/nvcomp.log"
  set +e
  NVCOMP_LOG_LEVEL=5 NVCOMP_LOG_FILE="${WORK}/nvcomp.log" timeout 15m python -m pytest -p no:cacheprovider \
    --numprocesses=8 --dist=worksteal -q . > "${WORK}/nvcomp-trace-${attempt}.log" 2>&1
  status=$?
  set -e
  crash=$(grep -o "crashed while running '[^']*'" "${WORK}/nvcomp-trace-${attempt}.log" | sort -u | tr '\n' ' ' || true)
  SUMMARY+=("nvcomp-trace attempt=${attempt} exit=${status} ${crash}")
  if [[ -n "${crash}" ]]; then
    rapids-logger "Attempt ${attempt} crashed: ${crash}"
    rapids-logger "nvCOMP messages that are not plain API traces"
    grep -v -E "\]\[Api\]\[" "${WORK}/nvcomp.log" | cut -c22- | sort | uniq -c | sort -rn | head -n 40 || true
    rapids-logger "Last nvCOMP calls per process"
    for pid in $(grep -oE "\]\[nvcomp\]\[[0-9]+\]" "${WORK}/nvcomp.log" | grep -oE "[0-9]+" | sort -u); do
      echo "=== pid ${pid}"
      grep -F "][nvcomp][${pid}]" "${WORK}/nvcomp.log" | grep -v "GetMaxOutputChunkSize" | tail -n 15 | cut -c1-600
    done
    break
  fi
done
popd > /dev/null

rapids-logger "GB300 nvCOMP trace summary"
printf '%s\n' "${SUMMARY[@]}"
