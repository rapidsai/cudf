#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the examples' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf/" || exit

# compute-sanitizer not available before CUDA 11.6
# Using -lt with decimals doesn't work in `bash` _except_ when comparing version strings
# shellcheck disable=SC2072
if [[ "${RAPIDS_CUDA_VERSION%.*}" < "11.6" ]]; then
  echo "computer-sanitizer unavailable pre 11.6"
  exit 0
fi

compute-sanitizer --tool memcheck basic_example

compute-sanitizer --tool memcheck deduplication

compute-sanitizer --tool memcheck custom_optimized names.csv
compute-sanitizer --tool memcheck custom_prealloc names.csv
compute-sanitizer --tool memcheck custom_with_malloc names.csv

compute-sanitizer --tool memcheck parquet_io example.parquet
compute-sanitizer --tool memcheck parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD TRUE

compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet
compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet 4 DEVICE_BUFFER 2 2

exit ${EXITCODE}
