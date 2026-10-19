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

apt-get update -qq && apt-get install -y -qq --no-install-recommends gdb > /dev/null

mapfile -t ALL_MODULES < <(find . -name "test_*.py" -not -path "./benchmarks/*" \
  -not -path "./test_types.py" -not -path "./test_json.py" | sed 's|^\./||' | sort)
NUM_SHARDS=8

for round in 1 2 3; do
  rapids-logger "Round ${round}: ${NUM_SHARDS} concurrent gdb-wrapped pytest shards"
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    modules=()
    for i in "${!ALL_MODULES[@]}"; do
      if (( (i + round) % NUM_SHARDS == shard )); then modules+=("${ALL_MODULES[$i]}"); fi
    done
    timeout 20m gdb -q -batch \
      -ex "handle SIGSEGV stop print nopass" \
      -ex "set pagination off" \
      -ex run \
      -ex "info sharedlibrary" \
      -ex "info threads" \
      -ex "thread apply all bt 40" \
      --args python -m pytest -p no:xdist -p no:cacheprovider -q "${modules[@]}" \
      > "${WORK}/round-${round}-shard-${shard}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done

  crashed=0
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    log="${WORK}/round-${round}-shard-${shard}.log"
    if grep -q "received signal SIGSEGV" "${log}"; then
      crashed=$((crashed + 1))
      if [[ ! -f "${WORK}/printed" ]]; then
        rapids-logger "Round ${round} shard ${shard} SIGSEGV: last tests, libraries, all-thread backtraces"
        grep -E "^(io/)?test_.*\.py [.sFxXE]" "${log}" | tail -n 3 || true
        sed -n '/received signal SIGSEGV/,$p' "${log}" \
          | grep -v -E "^\[(New Thread|Thread .* exited)" \
          | grep -v -E "^0x[0-9a-f]+ +0x[0-9a-f]+ +(Yes|No) .*(python3\.14/lib-dynload|site-packages/(numpy|pyarrow|cupy))" \
          | head -n 600 || true
        touch "${WORK}/printed"
      fi
    fi
  done
  SUMMARY+=("round=${round} crashed_shards=${crashed}/${NUM_SHARDS}")
  if [[ -f "${WORK}/printed" ]]; then break; fi
done

popd > /dev/null

rapids-logger "GB300 concurrent gdb summary"
printf '%s\n' "${SUMMARY[@]}"
