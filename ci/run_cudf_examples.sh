#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the examples' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf/" || exit

compute-sanitizer --tool memcheck basic_example

compute-sanitizer --tool memcheck deduplication

compute-sanitizer --tool memcheck custom_optimized names.csv
compute-sanitizer --tool memcheck custom_prealloc names.csv
compute-sanitizer --tool memcheck custom_with_malloc names.csv

compute-sanitizer --tool memcheck compute_checksum_jit info.csv output.csv
compute-sanitizer --tool memcheck extract_email_jit info.csv output.csv
compute-sanitizer --tool memcheck extract_email_precompiled info.csv output.csv
compute-sanitizer --tool memcheck format_phone_jit info.csv output.csv
compute-sanitizer --tool memcheck format_phone_precompiled info.csv output.csv
compute-sanitizer --tool memcheck localize_phone_jit info.csv output.csv
compute-sanitizer --tool memcheck localize_phone_precompiled info.csv output.csv

compute-sanitizer --tool memcheck parquet_io example.parquet
compute-sanitizer --tool memcheck parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD TRUE

compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet
compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet 4 DEVICE_BUFFER 2 2

exit ${EXITCODE}
