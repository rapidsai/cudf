# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame({"a": [1, 2, None], "b": [None, None, 5]}),
        pd.DataFrame(
            {"a": [1, 2, None], "b": [None, None, 5]}, index=["a", "p", "z"]
        ),
        pd.DataFrame({"a": [1, 2, 3]}),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        10,
        pd.Series([10, 20, 30]),
        pd.Series([3, 4, 5]),
        pd.Series([10, 20, 30], index=["z", "a", "p"]),
        {"a": 5, "b": pd.Series([3, 4, 5])},
        {"a": 5001},
        {"b": pd.Series([11, 22, 33], index=["a", "p", "z"])},
        {"a": 5, "b": pd.Series([3, 4, 5], index=["a", "p", "z"])},
        {"c": 100},
        np.nan,
    ],
)
def test_fillna_dataframe(pdf, value, inplace):
    if inplace:
        pdf = pdf.copy(deep=True)
    gdf = cudf.from_pandas(pdf)

    fill_value_pd = value
    if isinstance(fill_value_pd, (pd.Series, pd.DataFrame)):
        fill_value_cudf = cudf.from_pandas(fill_value_pd)
    elif isinstance(fill_value_pd, dict):
        fill_value_cudf = {}
        for key in fill_value_pd:
            temp_val = fill_value_pd[key]
            if isinstance(temp_val, pd.Series):
                temp_val = cudf.from_pandas(temp_val)
            fill_value_cudf[key] = temp_val
    else:
        fill_value_cudf = value

    expect = pdf.fillna(fill_value_pd, inplace=inplace)
    got = gdf.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = gdf
        expect = pdf

    assert_eq(expect, got)
