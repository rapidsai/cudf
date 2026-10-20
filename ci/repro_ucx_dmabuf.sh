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
export REPRO_GDB_CMDS="$ROOT/ci/repro_gdb.cmds"
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
    crashed=$(grep -l "Program received signal SIGSEGV" "$REPRO_LOG_DIR"/rank.*.log 2>/dev/null)
    if [ -n "$crashed" ]; then
      for f in $crashed; do
        echo "----- $f"
        python3 "$ROOT/ci/repro_report.py" "$f"
      done
      return 99
    fi
    if [ $r -ne 0 ]; then tail -12 "/tmp/seq.$tag.$nrank.log"; fi
  done
  return $rc
}

echo "=== plain sequences (1,2,5 ranks, exactly as CI)"
pf=0
for i in $(seq 1 5); do
  run_seq "plain$i" ""
  [ $? -ne 0 ] && pf=$((pf+1))
done
echo "plain sequence failures: $pf / 5"

echo "=== gdb sequences"
for i in $(seq 1 5); do
  run_seq "gdb$i" "$ROOT/ci/repro_gdb_rank.sh"
  [ $? -eq 99 ] && break
done
echo "repro script finished"
exit 0
