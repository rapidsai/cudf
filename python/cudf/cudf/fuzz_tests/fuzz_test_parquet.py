import sys

import pandas as pd

import cudf
from cudf.testing.main import pythonfuzz
from cudf.testing.parquet import ParquetReader, ParquetWriter
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=ParquetReader)
def parquet_reader_test(file_name):
    pdf = pd.read_parquet(file_name)
    gdf = cudf.read_parquet(file_name)

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
    if len(sys.argv) != 2:
        print("Usage is python file_name.py function_name")

    function_name_to_run = sys.argv[1]
    try:
        globals()[function_name_to_run]()
    except KeyError:
        print(
            f"Provided function name({function_name_to_run}) does not exist."
        )
