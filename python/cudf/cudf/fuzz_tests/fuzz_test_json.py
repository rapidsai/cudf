import sys

import pandas as pd

import cudf
from cudf.testing.json import JSONReader, JSONWriter
from cudf.testing.main import pythonfuzz
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=JSONReader)
def json_reader_test(file_name):
    pdf = pd.read_json(file_name)
    gdf = cudf.read_json(file_name)

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=JSONWriter)
def json_writer_test(gdf):
    pd_file_name = "cpu_pdf.json"
    gd_file_name = "gpu_pdf.json"

    pdf = gdf.to_pandas()

    pdf.to_json(pd_file_name)
    gdf.to_json(gd_file_name)

    actual = cudf.read_json(gd_file_name)
    expected = pd.read_json(pd_file_name)
    assert_eq(actual, expected)

    actual = cudf.read_json(pd_file_name)
    expected = pd.read_json(gd_file_name)
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
