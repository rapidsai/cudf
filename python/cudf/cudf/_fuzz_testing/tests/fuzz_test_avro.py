# Copyright (c) 2020, NVIDIA CORPORATION.

import sys

import cudf
from cudf._fuzz_testing.avro import AvroReader
from cudf._fuzz_testing.main import pythonfuzz
from cudf._fuzz_testing.utils import (
    ALL_POSSIBLE_VALUES,
    compare_dataframe,
    run_test,
)


@pythonfuzz(
    data_handle=AvroReader,
    params={
        "columns": ALL_POSSIBLE_VALUES,
        "skiprows": ALL_POSSIBLE_VALUES,
        "num_rows": ALL_POSSIBLE_VALUES,
    },
)
def avro_reader_test(input_tuple, columns, skiprows, num_rows):
    pdf, parquet_buffer = input_tuple
    expected_pdf = pdf[skiprows:]
    if num_rows is not None:
        expected_pdf = expected_pdf.head(num_rows)
    if skiprows is not None or num_rows is not None:
        expected_pdf = expected_pdf.reset_index(drop=True)

    gdf = cudf.read_avro(
        parquet_buffer, columns=columns, skiprows=skiprows, num_rows=num_rows
    )
    compare_dataframe(expected_pdf, gdf)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
