import sys

import pandas as pd

import cudf
from cudf.testing.csv import CSVReader, CSVWriter
from cudf.testing.main import pythonfuzz
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=CSVReader)
def csv_reader_test(file_name):
    print("csv_reader_test")
    pdf = pd.read_csv(file_name)
    gdf = cudf.read_csv(file_name)

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=CSVWriter)
def csv_writer_test(gdf):
    print("csv_writer_test")
    pd_file_name = "cpu_pdf.csv"
    gd_file_name = "gpu_pdf.csv"

    pdf = gdf.to_pandas()

    pdf.to_csv(pd_file_name)
    gdf.to_csv(gd_file_name)

    actual = cudf.read_csv(gd_file_name)
    expected = pd.read_csv(pd_file_name)
    assert_eq(actual, expected)

    actual = cudf.read_csv(pd_file_name)
    expected = pd.read_csv(gd_file_name)
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
