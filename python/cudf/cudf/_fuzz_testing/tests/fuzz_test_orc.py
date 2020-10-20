# Copyright (c) 2020, NVIDIA CORPORATION.

import sys

import cudf
from cudf._fuzz_testing.main import pythonfuzz
from cudf._fuzz_testing.orc import OrcReader
from cudf._fuzz_testing.utils import compare_dataframe, run_test


@pythonfuzz(
    data_handle=OrcReader,
    params={"columns": None, "skiprows": None, "num_rows": None},
)
def orc_reader_test(input_tuple, columns, num_rows):
    skiprows = 0
    pdf, parquet_buffer = input_tuple
    expected_pdf = pdf[skiprows:]
    if num_rows is not None:
        expected_pdf = expected_pdf.head(num_rows)
    if skiprows is not None or num_rows is not None:
        expected_pdf = expected_pdf.reset_index(drop=True)
    if columns is not None:
        expected_pdf = expected_pdf[columns]

    gdf = cudf.read_orc(
        parquet_buffer, columns=columns, skiprows=skiprows, num_rows=num_rows
    )

    compare_dataframe(expected_pdf, gdf)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
