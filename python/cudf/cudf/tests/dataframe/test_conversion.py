# Copyright (c) 2023, NVIDIA CORPORATION.
import pandas as pd

import cudf
from cudf.testing._utils import assert_eq


def test_convert_dtypes():
    data = {
        "a": [1, 2, 3],
        "b": [1, 2, 3],
        "c": [1.1, 2.2, 3.3],
        "d": [1.0, 2.0, 3.0],
        "e": [1.0, 2.0, 3.0],
        "f": ["a", "b", "c"],
        "g": ["a", "b", "c"],
        "h": ["2001-01-01", "2001-01-02", "2001-01-03"],
    }
    dtypes = [
        "int8",
        "int64",
        "float32",
        "float32",
        "float64",
        "str",
        "category",
        "datetime64[ns]",
    ]
    df = pd.DataFrame(
        {
            k: pd.Series(v, dtype=d)
            for k, v, d in zip(data.keys(), data.values(), dtypes)
        }
    )
    gdf = cudf.DataFrame.from_pandas(df)
    expect = df.convert_dtypes()
    got = gdf.convert_dtypes().to_pandas(nullable=True)
    assert_eq(expect, got)
