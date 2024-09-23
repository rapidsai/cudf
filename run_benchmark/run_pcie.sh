#!/usr/bin/env bash

parquet_reader_bench_bin=/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH
parquet_benchmark_name=parquet_read_io_compression

num_threads_arr=(8 1)

color_reset='\e[m'
color_green='\e[1;32m'

export KVIKIO_NTHREADS
export LIBCUDF_CUFILE_POLICY=OFF

# LIBCUDF_CUFILE_POLICY: Disable GDS
for num_threads in ${num_threads_arr[@]}; do
    echo -e "$color_green--> Thread count: $num_threads$color_reset"

    KVIKIO_NTHREADS=$num_threads
    ${parquet_reader_bench_bin} -d 0 -b $parquet_benchmark_name \
    -a compression_type=SNAPPY -a io_type=FILEPATH -a cardinality=0 -a run_length=1
done
