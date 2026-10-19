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

# Modules run by both crashing xdist workers before the crash, in gw7's order.
MODULES=(
  test_copying.py test_column_factories.py test_column_from_array.py test_column_from_device.py
  test_column_from_iterable.py test_column_span.py test_column_struct_from_children.py
  test_contiguous_split.py test_cuda_context.py test_datetime.py test_round.py test_experimental.py
  test_expressions.py test_filling.py test_gpumemoryview.py test_groupby.py test_hashing.py
  test_interop.py test_join.py test_labeling.py test_lists.py test_null_mask.py
  test_nvtext_byte_pair_encode.py test_nvtext_deduplicate.py test_nvtext_edit_distance.py
  test_nvtext_generate_ngrams.py test_nvtext_jaccard.py test_nvtext_minhash.py
  test_nvtext_ngrams_tokenize.py test_nvtext_normalize.py
)

run_serial() {
  local label=$1
  shift
  set +e
  CUDA_LAUNCH_BLOCKING=1 timeout 20m python -m pytest -p no:xdist -p no:cacheprovider -q "$@" io/test_types.py \
    > "${WORK}/${label}.log" 2>&1
  local status=$?
  set -e
  local crashed=""
  if grep -q "Fatal Python error" "${WORK}/${label}.log"; then crashed="SEGFAULT"; fi
  SUMMARY+=("${label} exit=${status} ${crashed}")
  if [[ -n "${crashed}" && ! -f "${WORK}/printed-stack" ]]; then
    rapids-logger "First crash stack (${label})"
    sed -n '/Fatal Python error/,/Extension modules/p' "${WORK}/${label}.log" | head -n 80 || true
    touch "${WORK}/printed-stack"
  fi
}

for attempt in 1 2; do
  run_serial "all-modules-${attempt}" "${MODULES[@]}"
done
for module in "${MODULES[@]}"; do
  for attempt in 1 2; do
    run_serial "${module}-${attempt}" "${module}"
  done
done

popd > /dev/null

rapids-logger "GB300 module bisect summary"
printf '%s\n' "${SUMMARY[@]}"
