#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd /home/coder/cudf

JOBS=${1:-$(nproc --ignore=2)}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMIT=$(git rev-parse --short HEAD)
TAG="${TIMESTAMP}_${COMMIT}"

echo "=== Build tag: ${TAG} ==="
echo "=== Jobs: ${JOBS} ==="

# Disable sccache, enable tests/benchmarks
echo "--- Configuring ---"
cmake -S cpp -B cpp/build/latest -G Ninja \
      -DCMAKE_C_COMPILER="$(which gcc)" \
      -DCMAKE_CXX_COMPILER="$(which g++)" \
      -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
      -DCMAKE_C_COMPILER_LAUNCHER="" \
      -DCMAKE_CXX_COMPILER_LAUNCHER="" \
      -DCMAKE_CUDA_COMPILER_LAUNCHER="" \
      -DCMAKE_CUDA_FLAGS="--threads=1" \
      -DCMAKE_CUDA_ARCHITECTURES=NATIVE \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=ON

# Verify sccache is disabled
if grep -q 'COMPILER_LAUNCHER:STRING=/' cpp/build/latest/CMakeCache.txt; then
    echo "ERROR: sccache is still enabled in CMakeCache.txt" >&2
    exit 1
fi

# Clean and rebuild
echo "--- Cleaning ---"
ninja -C cpp/build/latest clean

echo "--- Building (j${JOBS}) ---"
time ninja -C cpp/build/latest cudf -j"${JOBS}" 2>&1

# Save ninja log immediately after cudf build, before any additional targets
# (the sort_ninja_log.py parser resets on timestamp decreases from new ninja invocations)
mkdir -p compilation_reports
cp cpp/build/latest/.ninja_log "compilation_reports/${TAG}.ninja_log"

# Build stream usage test libraries (needed by ctest, not timed)
echo "--- Building stream usage test libs ---"
ninja -C cpp/build/latest \
  cudf_identify_stream_usage_mode_cudf \
  cudf_identify_stream_usage_mode_testing \
  -j"${JOBS}" 2>&1

# Generate reports from saved ninja log (not the live one which now includes test lib entries)
echo "--- Generating reports ---"

python cpp/scripts/sort_ninja_log.py "compilation_reports/${TAG}.ninja_log" --fmt csv \
  > "compilation_reports/${TAG}.csv"

python cpp/scripts/sort_ninja_log.py "compilation_reports/${TAG}.ninja_log" --fmt html \
  > "compilation_reports/${TAG}.html"

# Print top 20 slowest translation units
echo ""
echo "=== Top 20 slowest translation units ==="
sort -t',' -k1 -nr "compilation_reports/${TAG}.csv" | head -20

echo ""
echo "=== Reports saved ==="
echo "  CSV:       compilation_reports/${TAG}.csv"
echo "  HTML:      compilation_reports/${TAG}.html"
echo "  Ninja log: compilation_reports/${TAG}.ninja_log"
echo ""
echo "TAG=${TAG}"
