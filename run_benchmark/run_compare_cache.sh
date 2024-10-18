#!/usr/bin/env bash

parquet_reader_bench_bin=/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH
parquet_benchmark_name=parquet_read_io_compression

# parquet_benchmark_full_program=
parquet_benchmark_full_program="$parquet_reader_bench_bin -d 0 -b $parquet_benchmark_name
-a compression_type=SNAPPY -a io_type=FILEPATH -a cardinality=0 -a run_length=1 --min-samples 50"

color_reset='\e[m'
color_green='\e[1;32m'

#------------------------------------------------------------
# kvikIO setting
#------------------------------------------------------------
export KVIKIO_NTHREADS=1
export KVIKIO_COMPAT_MODE="off"

#------------------------------------------------------------
# cuDF setting
#------------------------------------------------------------
export LIBCUDF_CUFILE_POLICY="KVIKIO"

# Select an NVMe drive (/tmp is not)
export TMPDIR=/home/coder/cudf/run_benchmark

#------------------------------------------------------------
# cuFile setting
#------------------------------------------------------------
export CUFILE_ALLOW_COMPAT_MODE="false"
export CUFILE_LOGGING_LEVEL=WARN
export CUFILE_LOGFILE_PATH


setup="${LIBCUDF_CUFILE_POLICY}_kvik_compat_${KVIKIO_COMPAT_MODE}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${KVIKIO_NTHREADS}_cold"
CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"
echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
CUDF_BENCHMARK_DROP_CACHE=true $parquet_benchmark_full_program


setup="${LIBCUDF_CUFILE_POLICY}_kvik_compat_${KVIKIO_COMPAT_MODE}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${KVIKIO_NTHREADS}_hot"
CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"
echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
$parquet_benchmark_full_program
