#!/usr/bin/env bash

# References
# https://developer.nvidia.com/blog/boosting-data-ingest-throughput-with-gpudirect-storage-and-rapids-cudf/

# Drop the cache for the benchmark
export CUDF_BENCHMARK_DROP_CACHE=true

# Specify GDS behavior
export LIBCUDF_CUFILE_POLICY

# Default axis set in the bench:
# compression_type: "SNAPPY", "NONE"
# cardinality: 0, 1000
# run_length: 1, 32
parquet_reader_bench_bin=${HOME}/cudf/cpp/build/latest/benchmarks/PARQUET_READER_NVBENCH
parquet_benchmark_name=parquet_read_io_compression

# Default axis set in the bench:
# compression_type: "SNAPPY", "NONE"
# cardinality: 0, 1000
# run_length: 1, 32
orc_reader_bench_bin=${HOME}/cudf/cpp/build/latest/benchmarks/ORC_READER_NVBENCH
orc_benchmark_name=orc_read_io_compression



enable_gds=(false true)

for enable_gds_status in ${enable_gds[@]}; do
    output_suffix=""

    if [ "$enable_gds_status" = true ]; then
        output_suffix="gds_enabled"
        LIBCUDF_CUFILE_POLICY=GDS
    else
        output_suffix="gds_disabled"
        LIBCUDF_CUFILE_POLICY=OFF
    fi

    ${parquet_reader_bench_bin} -d 0 -b $parquet_benchmark_name -a io_type=FILEPATH -a cardinality=0 -a run_length=1 --csv "parquet_reader_bench_$output_suffix.csv" --stopping-criterion entropy

    ${orc_reader_bench_bin} -d 0 -b $orc_benchmark_name -a io=FILEPATH -a cardinality=0 -a run_length=1 --csv "orc_reader_bench_$output_suffix.csv" --stopping-criterion entropy
done

