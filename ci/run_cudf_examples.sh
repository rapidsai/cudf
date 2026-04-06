#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR

# Support customizing the examples' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libcudf" || exit

pushd basic || exit
compute-sanitizer --tool memcheck basic_example
popd || exit

GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')
CSE="compute-sanitizer"
if [[ "${GPU_COMPUTE_CAP}" == "12.0" ]]; then
    CSE=""
fi

pushd hybrid_scan_io || exit
echo hybrid_scan_io
${CSE} ./hybrid_scan_io example.parquet string_col 0000001 PINNED_BUFFER
${CSE} ./hybrid_scan_pipeline example.parquet 2 HOST_BUFFER ROW_GROUPS 2
${CSE} ./hybrid_scan_pipeline example.parquet 2 FILEPATH BYTE_RANGES 2
${CSE} ./hybrid_scan_multifile_single_step example.parquet 10 2 YES DEVICE_BUFFER 2
${CSE} ./hybrid_scan_multifile_single_step example.parquet 10 2 NO FILEPATH 1
${CSE} ./hybrid_scan_multifile_two_step example.parquet 10 2 string_col 0000001 PINNED_BUFFER 2
${CSE} ./hybrid_scan_multifile_two_step example.parquet 10 2 string_col 0000001 HOST_BUFFER 1
popd || exit

pushd nested_types || exit
echo deduplication
${CSE} ./deduplication
popd || exit

pushd parquet_io || exit
echo parquet_io
${CSE} ./parquet_io example.parquet
${CSE} ./parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD TRUE
${CSE} ./parquet_io_multithreaded example.parquet
${CSE} ./parquet_io_multithreaded example.parquet 4 DEVICE_BUFFER 2 2
popd || exit

pushd parquet_inspect || exit
echo parquet_inspect
${CSE} ./parquet_inspect example.parquet
popd || exit

pushd strings || exit
echo strings
${CSE} ./custom_optimized names.csv
${CSE} ./custom_prealloc names.csv
${CSE} ./custom_with_malloc names.csv
popd || exit

pushd string_transformers || exit
echo string_transformers
${CSE} ./compute_checksum_jit info.csv output.csv
${CSE} ./extract_email_jit info.csv output.csv
${CSE} ./extract_email_precompiled info.csv output.csv
${CSE} ./format_phone_jit info.csv output.csv
${CSE} ./format_phone_precompiled info.csv output.csv
${CSE} ./localize_phone_jit info.csv output.csv
${CSE} ./localize_phone_precompiled info.csv output.csv
popd || exit

exit ${EXITCODE}
