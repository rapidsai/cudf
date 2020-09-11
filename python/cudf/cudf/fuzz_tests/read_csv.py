import pandas as pd

import cudf
from cudf.testing.csv import CSVReader
from cudf.testing.main import pythonfuzz
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=CSVReader)
def csv_reader_test(file_name):
    pdf = pd.read_csv(file_name)
    gdf = cudf.read_csv(file_name)

    assert_eq(gdf, pdf)


if __name__ == "__main__":
    csv_reader_test()
