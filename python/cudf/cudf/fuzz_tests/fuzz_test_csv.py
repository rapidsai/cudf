# Copyright (c) 2020, NVIDIA CORPORATION.

import sys
from io import StringIO

import pandas as pd

import cudf
from cudf.testing.csv import CSVReader, CSVWriter
from cudf.testing.main import pythonfuzz
from cudf.testing.utils import compare_content, run_test
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=CSVReader)
def csv_reader_test(csv_buffer):
    pdf = pd.read_csv(csv_buffer)
    gdf = cudf.read_csv(csv_buffer)

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=CSVWriter)
def csv_writer_test(gdf):
    pdf = gdf.to_pandas()

    pd_buffer = pdf.to_csv()
    gd_buffer = gdf.to_csv()

    compare_content(pd_buffer, gd_buffer)

    actual = cudf.read_csv(StringIO(gd_buffer))
    expected = pd.read_csv(StringIO(pd_buffer))
    assert_eq(actual, expected)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
