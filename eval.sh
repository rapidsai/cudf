#!/bin/bash
# CSV Reader Autoresearch Eval Script
# Runs 3 target benchmarks + NVTX stage profiling.
# Usage: ./eval.sh [results_dir]

set -euo pipefail

BENCH_DIR="${BENCH_DIR:-cpp/build/latest/benchmarks}"
RESULTS_DIR="${1:-results/$(date +%m%d_%H%M%S)}"
TIMEOUT=5

mkdir -p "$RESULTS_DIR"

echo "=== CSV Reader Eval ==="
echo "Benchmark dir: $BENCH_DIR"
echo "Results dir:   $RESULTS_DIR"
echo ""

# 3 target benchmarks
for name in CSV_READER_REALISTIC_NVBENCH CSV_READER_TYPE_INFERENCE_NVBENCH CSV_READER_QUOTING_NVBENCH; do
  echo "Running $name ..."
  "$BENCH_DIR/$name" --timeout "$TIMEOUT" --json "$RESULTS_DIR/$name.json"
  echo ""
done

# NVTX stage profiling on TAXI/256MB
echo "=== NVTX Stage Profile (TAXI/256MB) ==="
NSYS_REPORT="$RESULTS_DIR/nvtx_profile"
nsys profile --trace=nvtx --nvtx-domain-include=libcudf \
  --force-overwrite=true -o "$NSYS_REPORT" \
  "$BENCH_DIR/CSV_READER_REALISTIC_NVBENCH" \
  --profile --axis "profile=TAXI" --axis "data_size_mb=256" \
  > /dev/null 2>&1

nsys stats --report nvtx_pushpop_sum --format column "${NSYS_REPORT}.nsys-rep" \
  --output @"grep -E (-|csv::|read_csv)" --force-export=true \
  2>&1 | tee "$RESULTS_DIR/nvtx_stages.txt"

echo ""
echo "Results saved to $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
