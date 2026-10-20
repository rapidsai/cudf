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
    crashed=$(grep -l "Program received signal SIGSEGV" "$REPRO_LOG_DIR"/rank.*.log 2>/dev/null)
    if [ -n "$crashed" ]; then
      for f in $crashed; do
        echo "----- $f"
        python3 "$ROOT/ci/repro_report.py" "$f"
      done
      return 99
    fi
    if [ $r -ne 0 ]; then
      if [ -f "$REPRO_LOG_DIR/rank.0.log" ]; then echo "----- rank.0.log (head)"; head -20 "$REPRO_LOG_DIR/rank.0.log"; fi
      echo "----- crash output ($tag, $nrank ranks)"
      grep -v -E "^\s*$|platform linux|rootdir|configfile|plugins|test session|collect" "/tmp/seq.$tag.$nrank.log" | tail -30
    fi
  done
  return $rc
}

echo "=== gdb sanity"
export REPRO_GDB_CMDS="$ROOT/ci/repro_gdb.cmds"
export REPRO_LOG_DIR=/tmp/gdbsanity
mkdir -p "$REPRO_LOG_DIR"
gdb --version | head -1
cat /proc/sys/kernel/yama/ptrace_scope 2>&1
gdb -batch -x "$REPRO_GDB_CMDS" -ex run -ex 'bt 3' --args python -c "import ctypes; ctypes.CDLL('libcuda.so.1').cuInit(0); print('hello')" 2>&1 | tail -15

echo "=== gdb sequences"
for i in $(seq 1 10); do
  run_seq "gdb$i" "$ROOT/ci/repro_gdb_rank.sh"
  if [ $? -eq 99 ]; then break; fi
done
echo "repro script finished"
exit 0
