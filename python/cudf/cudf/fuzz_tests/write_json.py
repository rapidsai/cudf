import pandas as pd

import cudf
from cudf.testing.json import JSONWriter
from cudf.testing.main import pythonfuzz
from cudf.tests.utils import assert_eq


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
    json_writer_test()
