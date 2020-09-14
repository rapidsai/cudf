import pandas as pd

import cudf
from cudf.testing.json import JSONReader
from cudf.testing.main import pythonfuzz
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=JSONReader)
def json_reader_test(file_name):
    pdf = pd.read_json(file_name)
    gdf = cudf.read_json(file_name)

    assert_eq(gdf, pdf)


if __name__ == "__main__":
    json_reader_test()
