# Copyright (c) 2020, NVIDIA CORPORATION.

import sys

import cudf
from cudf._fuzz_testing.avro import AvroReader
from cudf._fuzz_testing.main import pythonfuzz
from cudf._fuzz_testing.utils import run_test
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=AvroReader)
def avro_reader_test(input_tuple):
    pdf, parquet_buffer = input_tuple
    gdf = cudf.read_avro(parquet_buffer)
    assert_eq(pdf, gdf)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
