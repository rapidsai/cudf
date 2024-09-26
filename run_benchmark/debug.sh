#!/usr/bin/env bash

parquet_reader_bench_bin=/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH
parquet_benchmark_name=parquet_read_io_compression

color_reset='\e[m'
color_green='\e[1;32m'

# Force all CUDA calls to be synchronous

# Only allow GDS IO. Disallow POSIX fallback.
# export CUFILE_ALLOW_COMPAT_MODE=false

export KVIKIO_NTHREADS=1

export LIBCUDF_CUFILE_POLICY="GDS"

# export KVIKIO_COMPAT_MODE=on

export LIBCUDF_CUFILE_THREAD_COUNT=1

export TMPDIR=/home/coder/cudf/run_benchmark

CUFILE_LOGGING_LEVEL=TRACE

gdb -ex start --args ${parquet_reader_bench_bin} -d 0 -b $parquet_benchmark_name \
-a compression_type=NONE -a io_type=FILEPATH -a cardinality=0 -a run_length=1 \
--run-once

# ${parquet_reader_bench_bin} -d 0 -b $parquet_benchmark_name \
# -a compression_type=NONE -a io_type=FILEPATH -a cardinality=0 -a run_length=1 \
# --run-once

