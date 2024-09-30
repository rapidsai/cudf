#!/usr/bin/env bash

parquet_reader_bench_bin=/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH
parquet_benchmark_name=parquet_read_io_compression

color_reset='\e[m'
color_green='\e[1;32m'

#------------------------------------------------------------
# kvikIO setting
#------------------------------------------------------------
num_threads_arr=(8 1)
export KVIKIO_NTHREADS

export KVIKIO_COMPAT_MODE=off

#------------------------------------------------------------
# cuDF setting
#------------------------------------------------------------
cufile_policy_arr=("GDS" "ALWAYS" "KVIKIO" "OFF")
export LIBCUDF_CUFILE_POLICY

# Drop the cache for the benchmark
# export CUDF_BENCHMARK_DROP_CACHE=true

# Select an NVMe drive (/tmp is not)
export TMPDIR=/home/coder/cudf/run_benchmark

#------------------------------------------------------------
# cuFile setting
#------------------------------------------------------------
# Only allow GDS IO. Disallow POSIX fallback.
export CUFILE_ALLOW_COMPAT_MODE=false

export CUFILE_LOGGING_LEVEL=WARN

export CUFILE_LOGFILE_PATH



for cufile_policy in ${cufile_policy_arr[@]}; do
    LIBCUDF_CUFILE_POLICY=$cufile_policy

    for num_threads in ${num_threads_arr[@]}; do
        KVIKIO_NTHREADS=$num_threads
        CUFILE_LOGFILE_PATH="cufile_log_${cufile_policy}_${num_threads}.txt"
        echo -e "$color_green--> Thread count: $num_threads, cuFile policy: $cufile_policy$color_reset"
        ${parquet_reader_bench_bin} -d 0 -b $parquet_benchmark_name \
        -a compression_type=SNAPPY -a io_type=FILEPATH -a cardinality=0 -a run_length=1
    done
done





