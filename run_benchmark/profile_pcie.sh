#!/usr/bin/env bash

# Run this script with sudo in order to collect performance counters

nsys_bin=/mnt/nsight-systems-cli/2024.5.1/bin/nsys
parquet_reader_bench_bin=/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH
parquet_benchmark_name=parquet_read_io_compression

num_threads_arr=(8 1)

color_reset='\e[m'
color_green='\e[1;32m'

for num_threads in ${num_threads_arr[@]}; do
    echo -e "$color_green--> Thread count: $num_threads$color_reset"

    $nsys_bin profile \
    -o /mnt/profile/$num_threads \
    -t nvtx,cuda,osrt \
    -f true \
    --backtrace=none \
    --gpu-metrics-devices=0 \
    --gpuctxsw=true \
    --cuda-memory-usage=true \
    --env-var KVIKIO_NTHREADS=$num_threads,LIBCUDF_CUFILE_POLICY=OFF \
    ${parquet_reader_bench_bin} -d 0 -b $parquet_benchmark_name \
    -a compression_type=SNAPPY -a io_type=FILEPATH -a cardinality=0 -a run_length=1 --timeout 1 --run-once
done
