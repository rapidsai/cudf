#!/usr/bin/env bash

parquet_reader_bench_bin=/home/coder/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH
parquet_benchmark_name=parquet_read_io_compression

# parquet_benchmark_full_program=
parquet_benchmark_full_program="$parquet_reader_bench_bin -d 0 -b $parquet_benchmark_name
-a compression_type=SNAPPY -a io_type=FILEPATH -a cardinality=0 -a run_length=1"

color_reset='\e[m'
color_green='\e[1;32m'

#------------------------------------------------------------
# kvikIO setting
#------------------------------------------------------------
num_threads_arr=(1 8)
export KVIKIO_NTHREADS

export KVIKIO_COMPAT_MODE

#------------------------------------------------------------
# cuDF setting
#------------------------------------------------------------
export LIBCUDF_CUFILE_POLICY

# Drop the cache for the benchmark
export CUDF_BENCHMARK_DROP_CACHE=true

# Select an NVMe drive (/tmp is not)
export TMPDIR=/home/coder/cudf/run_benchmark

#------------------------------------------------------------
# cuFile setting
#------------------------------------------------------------
export CUFILE_ALLOW_COMPAT_MODE
export CUFILE_FORCE_COMPAT_MODE

export CUFILE_LOGGING_LEVEL=WARN

export CUFILE_LOGFILE_PATH


# GDS
LIBCUDF_CUFILE_POLICY="GDS"
CUFILE_ALLOW_COMPAT_MODE="false"
for num_threads in ${num_threads_arr[@]}; do
    KVIKIO_NTHREADS=$num_threads
    setup="${LIBCUDF_CUFILE_POLICY}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done

LIBCUDF_CUFILE_POLICY="GDS"
CUFILE_ALLOW_COMPAT_MODE="true"
CUFILE_FORCE_COMPAT_MODE="true"
for num_threads in ${num_threads_arr[@]}; do
    KVIKIO_NTHREADS=$num_threads
    setup="${LIBCUDF_CUFILE_POLICY}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done

# ALWAYS
LIBCUDF_CUFILE_POLICY="ALWAYS"
CUFILE_ALLOW_COMPAT_MODE="false"
for num_threads in ${num_threads_arr[@]}; do
    KVIKIO_NTHREADS=$num_threads
    setup="${LIBCUDF_CUFILE_POLICY}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done

LIBCUDF_CUFILE_POLICY="ALWAYS"
CUFILE_ALLOW_COMPAT_MODE="true"
CUFILE_FORCE_COMPAT_MODE="true"
for num_threads in ${num_threads_arr[@]}; do
    KVIKIO_NTHREADS=$num_threads
    setup="${LIBCUDF_CUFILE_POLICY}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done

# KVIKIO
LIBCUDF_CUFILE_POLICY="KVIKIO"
KVIKIO_COMPAT_MODE="off"
CUFILE_ALLOW_COMPAT_MODE="false"
for num_threads in ${num_threads_arr[@]}; do
    KVIKIO_NTHREADS=$num_threads
    setup="${LIBCUDF_CUFILE_POLICY}_kvik_compat_${KVIKIO_COMPAT_MODE}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done

LIBCUDF_CUFILE_POLICY="KVIKIO"
KVIKIO_COMPAT_MODE="off"
CUFILE_ALLOW_COMPAT_MODE="true"
CUFILE_FORCE_COMPAT_MODE="true"
for num_threads in ${num_threads_arr[@]}; do
    KVIKIO_NTHREADS=$num_threads
    setup="${LIBCUDF_CUFILE_POLICY}_kvik_compat_${KVIKIO_COMPAT_MODE}_cufile_compat_${CUFILE_ALLOW_COMPAT_MODE}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done

LIBCUDF_CUFILE_POLICY="KVIKIO"
KVIKIO_COMPAT_MODE="on"
for num_threads in ${num_threads_arr[@]}; do
    KVIKIO_NTHREADS=$num_threads
    setup="${LIBCUDF_CUFILE_POLICY}_kvik_compat_${KVIKIO_COMPAT_MODE}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done


# OFF
LIBCUDF_CUFILE_POLICY="OFF"
for num_threads in ${num_threads_arr[@]}; do
    setup="${LIBCUDF_CUFILE_POLICY}_${num_threads}"
    CUFILE_LOGFILE_PATH="cufile_log_${setup}.txt"

    echo -e "$color_green--> Thread count: $num_threads, setup: $setup$color_reset"
    $parquet_benchmark_full_program
done
