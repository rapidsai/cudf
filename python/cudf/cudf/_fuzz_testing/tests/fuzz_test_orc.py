# Copyright (c) 2020, NVIDIA CORPORATION.

import sys

import cudf
from cudf._fuzz_testing.main import pythonfuzz
from cudf._fuzz_testing.orc import OrcReader
from cudf._fuzz_testing.utils import run_test
from cudf.tests.utils import assert_eq


@pythonfuzz(
    data_handle=OrcReader,
    params={"columns": None, "skiprows": None, "num_rows": None},
)
def orc_reader_test(input_tuple, columns, skiprows, num_rows):
    pdf, parquet_buffer = input_tuple
    expected_pdf = pdf[skiprows:]
    if num_rows is not None:
        expected_pdf = expected_pdf.head(num_rows)
    if skiprows is not None or num_rows is not None:
        expected_pdf = expected_pdf.reset_index(drop=True)

    gdf = cudf.read_orc(
        parquet_buffer, columns=columns, skiprows=skiprows, num_rows=num_rows
    )
    assert_eq(expected_pdf, gdf)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
