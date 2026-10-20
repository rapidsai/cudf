#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1
source ./ci/test_python_common.sh test_python_other
set +e

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1
ROOT="$PWD"
TESTDIR="$PWD/python/cudf_streaming/cudf_streaming/tests"
cd "$TESTDIR" || exit 1

PYTEST_ARGS=(--cache-clear --junitxml="/tmp/junit.xml" --durations=50 --durations-min=1 .)

run_seq() {
  local tag="$1" wrapper="$2" rc=0
  for nrank in 1 2 5; do
    export REPRO_LOG_DIR="/tmp/gdb.$tag.$nrank"
    mkdir -p "$REPRO_LOG_DIR"
    # shellcheck disable=SC2086
    timeout 600 mpirun --oversubscribe --map-by node --bind-to none -np "$nrank" \
      $wrapper python -m pytest "${PYTEST_ARGS[@]}" > "/tmp/seq.$tag.$nrank.log" 2>&1
    r=$?
    echo "seq $tag nranks=$nrank rc=$r"
    [ $r -ne 0 ] && rc=$r
    if [ $r -ne 0 ]; then
      echo "----- crash output ($tag, $nrank ranks)"
      grep -v -E "^\s*$|^\S*\.\S*\s*\[" "/tmp/seq.$tag.$nrank.log" | tail -80
    fi
  done
  return $rc
}

echo "=== compilers"
which cc gcc g++ clang || echo "no compiler"

echo "=== driver race"
python "$ROOT/ci/repro_race.py"

echo "=== sequences with UCX native crash handler"
export UCX_HANDLE_ERRORS=bt
PYTEST_ARGS=(-p no:faulthandler "${PYTEST_ARGS[@]}")
pf=0
for i in $(seq 1 8); do
  run_seq "ucxbt$i" ""
  [ $? -ne 0 ] && pf=$((pf+1))
done
echo "sequence failures: $pf / 8"
echo "repro script finished"
exit 0
