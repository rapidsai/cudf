#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the examples' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf/" || exit

compute-sanitizer --tool memcheck basic/basic_example

compute-sanitizer --tool memcheck nested_types/deduplication

compute-sanitizer --tool memcheck strings/custom_optimized strings/names.csv
compute-sanitizer --tool memcheck strings/custom_prealloc strings/names.csv
compute-sanitizer --tool memcheck strings/custom_with_malloc strings/names.csv

cd string_transformers
compute-sanitizer --tool memcheck branching_public info.csv output.csv
compute-sanitizer --tool memcheck branching info.csv output.csv
compute-sanitizer --tool memcheck int_output info.csv output.csv
compute-sanitizer --tool memcheck output_public info.csv output.csv
compute-sanitizer --tool memcheck output info.csv output.csv
compute-sanitizer --tool memcheck preallocated_public info.csv output.csv
compute-sanitizer --tool memcheck preallocated info.csv output.csv
cd ..

cd parquet_io
compute-sanitizer --tool memcheck parquet_io example.parquet
compute-sanitizer --tool memcheck parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD TRUE

compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet
compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet 4 DEVICE_BUFFER 2 2
cd ..

exit ${EXITCODE}
