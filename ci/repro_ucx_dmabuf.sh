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

NRANKS="${REPRO_NRANKS:-5}"
TEST="${REPRO_TEST:-test_streaming_allgather.py}"

echo "=== plain runs: $NRANKS ranks, $TEST"
fails=0
for i in $(seq 1 8); do
  timeout 300 mpirun --oversubscribe --map-by node --bind-to none -np "$NRANKS" \
    python -m pytest --cache-clear -x -q "$TEST" > "/tmp/plain.$i.log" 2>&1
  rc=$?
  echo "plain run $i rc=$rc"
  [ $rc -ne 0 ] && fails=$((fails+1))
  if [ $rc -ne 0 ]; then tail -25 "/tmp/plain.$i.log"; fi
done
echo "plain failures: $fails / 8"

echo "=== gdb runs"
for i in $(seq 1 12); do
  export REPRO_LOG_DIR="/tmp/gdb.$i"
  mkdir -p "$REPRO_LOG_DIR"
  timeout 600 mpirun --oversubscribe --map-by node --bind-to none -np "$NRANKS" \
    "$ROOT/ci/repro_gdb_rank.sh" python -m pytest --cache-clear -x -q "$TEST" > "$REPRO_LOG_DIR/mpirun.log" 2>&1
  rc=$?
  echo "gdb run $i rc=$rc"
  crashed=$(grep -l "STOPPED signal" "$REPRO_LOG_DIR"/rank.*.log 2>/dev/null)
  if [ -n "$crashed" ]; then
    for f in $crashed; do
      echo "----- $f (last GETHANDLE calls, then crash dump)"
      grep -c GETHANDLE "$f"
      grep GETHANDLE "$f" | tail -5
      sed -n '/STOPPED signal/,/Start Addr/p' "$f" | head -80
      grep -E "libcuda|libucp|libucs|libuct" "$f" | tail -8
    done
    break
  fi
done
echo "repro script finished"
exit 0
