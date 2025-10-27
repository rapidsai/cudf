#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the examples' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf" || exit

pushd basic || exit
compute-sanitizer --tool memcheck basic_example
popd || exit

pushd hybrid_scan_io || exit
compute-sanitizer --tool memcheck hybrid_scan_io example.parquet string_col 0000001  PINNED_BUFFER
popd || exit

pushd nested_types || exit
compute-sanitizer --tool memcheck deduplication
popd || exit

pushd parquet_io || exit
compute-sanitizer --tool memcheck parquet_io example.parquet
compute-sanitizer --tool memcheck parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD TRUE

compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet
compute-sanitizer --tool memcheck parquet_io_multithreaded example.parquet 4 DEVICE_BUFFER 2 2
popd || exit

pushd parquet_inspect || exit
compute-sanitizer --tool memcheck parquet_inspect example.parquet
popd || exit

pushd strings || exit
compute-sanitizer --tool memcheck custom_optimized names.csv
compute-sanitizer --tool memcheck custom_prealloc names.csv
compute-sanitizer --tool memcheck custom_with_malloc names.csv
popd || exit

pushd string_transformers || exit
compute-sanitizer --tool memcheck compute_checksum_jit info.csv output.csv
compute-sanitizer --tool memcheck extract_email_jit info.csv output.csv
compute-sanitizer --tool memcheck extract_email_precompiled info.csv output.csv
compute-sanitizer --tool memcheck format_phone_jit info.csv output.csv
compute-sanitizer --tool memcheck format_phone_precompiled info.csv output.csv
compute-sanitizer --tool memcheck localize_phone_jit info.csv output.csv
compute-sanitizer --tool memcheck localize_phone_precompiled info.csv output.csv
popd || exit

exit ${EXITCODE}
