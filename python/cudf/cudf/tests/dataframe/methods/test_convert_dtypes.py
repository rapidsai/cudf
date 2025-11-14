# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


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
    nullable_columns = list("abcdef")
    non_nullable_columns = list(set(data.keys()).difference(nullable_columns))

    df = pd.DataFrame(
        {
            k: pd.Series(v, dtype=d)
            for k, v, d in zip(data.keys(), data.values(), dtypes, strict=True)
        }
    )
    gdf = cudf.DataFrame(df)
    expect = df[nullable_columns].convert_dtypes()
    got = gdf[nullable_columns].convert_dtypes().to_pandas(nullable=True)
    assert_eq(expect, got)

    with pytest.raises(NotImplementedError):
        # category and datetime64[ns] are not nullable
        gdf[non_nullable_columns].convert_dtypes().to_pandas(nullable=True)
