#!/usr/bin/env bash

root_dir=/home/coder/cudf/cpp/examples/parquet_io

example_bin=$root_dir/build/parquet_io
input_file=$root_dir/example.parquet
output_file=$root_dir/debug/test_output.parquet

# DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY
encoding_type=DEFAULT

# NONE, AUTO, SNAPPY, LZ4, ZSTD
compression_type=NONE

write_page_stats=yes

export LIBCUDF_LOGGING_LEVEL=INFO

# cuda-gdb -ex start --args $example_bin $input_file $output_file $encoding_type $compression_type

$example_bin $input_file $output_file $encoding_type $compression_type $write_page_stats

