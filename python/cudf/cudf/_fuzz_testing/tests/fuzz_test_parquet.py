# Copyright (c) 2020, NVIDIA CORPORATION.

import sys

import numpy as np
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


@pythonfuzz(
    data_handle=ParquetReader,
    params={"columns": None, "use_pandas_metadata": [True, False]},
)
def parquet_reader_columns(parquet_buffer, columns, use_pandas_metadata):
    pdf = pd.read_parquet(
        parquet_buffer,
        columns=columns,
        use_pandas_metadata=use_pandas_metadata,
    )
    gdf = cudf.read_parquet(
        parquet_buffer,
        columns=columns,
        use_pandas_metadata=use_pandas_metadata,
    )

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=ParquetWriter)
def parquet_writer_test(pdf):
    pd_file_name = "cpu_pdf.parquet"
    gd_file_name = "gpu_pdf.parquet"

    gdf = cudf.from_pandas(pdf)

    pdf.to_parquet(pd_file_name)
    gdf.to_parquet(gd_file_name)

    actual = cudf.read_parquet(gd_file_name)
    expected = pd.read_parquet(pd_file_name)
    assert_eq(actual, expected)

    actual = cudf.read_parquet(pd_file_name)
    expected = pd.read_parquet(gd_file_name)
    assert_eq(actual, expected)


@pythonfuzz(
    data_handle=ParquetWriter,
    params={
        "row_group_size": np.random.random_integers(0, 10000, 100),
        "compression": ["snappy", None],
    },
)
def parquet_writer_test_rowgroup_index_compression(
    pdf, compression, row_group_size
):
    pd_file_name = "cpu_pdf.parquet"
    gd_file_name = "gpu_pdf.parquet"

    gdf = cudf.from_pandas(pdf)

    pdf.to_parquet(
        pd_file_name, compression=compression, row_group_size=row_group_size,
    )
    gdf.to_parquet(
        gd_file_name, compression=compression, row_group_size=row_group_size,
    )

    actual = cudf.read_parquet(gd_file_name)
    expected = pd.read_parquet(pd_file_name)
    assert_eq(actual, expected)

    actual = cudf.read_parquet(pd_file_name)
    expected = pd.read_parquet(gd_file_name)
    assert_eq(actual, expected)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
