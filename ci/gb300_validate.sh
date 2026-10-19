#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

LIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBCUDF_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

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
  [[ -z "${crash}" ]]
}

pushd python/pylibcudf/tests > /dev/null
for attempt in 1 2 3 4 5 6; do
  run_suite "fix-${attempt}" || FIX_CRASHES=$((FIX_CRASHES + 1))
done
popd > /dev/null

rapids-logger "GB300 fix validation summary"
printf '%s\n' "${SUMMARY[@]}"
echo "Runs with crashes: ${FIX_CRASHES}/6"
[[ ${FIX_CRASHES} -eq 0 ]]
