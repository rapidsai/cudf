# Copyright (c) 2020, NVIDIA CORPORATION.

import sys

import pandas as pd

import cudf
from cudf.testing.main import pythonfuzz
from cudf.testing.parquet import ParquetReader, ParquetWriter
from cudf.testing.utils import run_test
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=ParquetReader)
def parquet_reader_test(parquet_buffer):
    pdf = pd.read_parquet(parquet_buffer)
    gdf = cudf.read_parquet(parquet_buffer)

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=ParquetWriter)
def parquet_writer_test(gdf):
    pd_file_name = "cpu_pdf.parquet"
    gd_file_name = "gpu_pdf.parquet"

    pdf = gdf.to_pandas()

    pdf.to_parquet(pd_file_name)
    gdf.to_parquet(gd_file_name)

    actual = cudf.read_parquet(gd_file_name)
    expected = pd.read_parquet(pd_file_name)
    assert_eq(actual, expected)

    actual = cudf.read_parquet(pd_file_name)
    expected = pd.read_parquet(gd_file_name)
    assert_eq(actual, expected)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
