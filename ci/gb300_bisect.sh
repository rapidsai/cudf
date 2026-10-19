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

# A/B the test_interop.py change with everything else identical.
mkdir -p "${WORK}/unfixed" "${WORK}/fixed"
cp -r python/pylibcudf/tests "${WORK}/fixed/tests"
cp -r python/pylibcudf/tests "${WORK}/unfixed/tests"
git show 4473c24036cba0af26b11730e750782f3fc740ed:python/pylibcudf/tests/test_interop.py \
  > "${WORK}/unfixed/tests/test_interop.py"

SUMMARY=()
for variant in unfixed:6 fixed:10; do
  label="${variant%%:*}"
  count="${variant##*:}"
  pushd "${WORK}/${label}/tests" > /dev/null
  for attempt in $(seq "${count}"); do
    set +e
    timeout 15m python -m pytest -p no:cacheprovider --numprocesses=8 --dist=worksteal -q . \
      > "${WORK}/${label}-${attempt}.log" 2>&1
    status=$?
    set -e
    tail -n 1 "${WORK}/${label}-${attempt}.log"
    crash=$(grep -o "crashed while running '[^']*'" "${WORK}/${label}-${attempt}.log" | sort -u | tr '\n' ' ' || true)
    SUMMARY+=("${label} attempt=${attempt} exit=${status} ${crash}")
  done
  popd > /dev/null
done

rapids-logger "GB300 test_interop fix A/B summary"
printf '%s\n' "${SUMMARY[@]}"
