import pandas as pd

import cudf
from cudf.testing.main import PythonFuzz
from cudf.tests.utils import assert_eq


@PythonFuzz
def parquet_reader_test(file_name):
    pdf = pd.read_parquet(file_name)
    gdf = cudf.read_parquet(file_name)

    assert_eq(gdf, pdf)


if __name__ == "__main__":
    parquet_reader_test()
