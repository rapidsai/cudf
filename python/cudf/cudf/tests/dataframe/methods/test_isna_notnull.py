# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "na_data",
    [
        pd.DataFrame(
            {
                "a": [0, 1, 2, np.nan, 4, None, 6],
                "b": [np.nan, None, "u", "h", "d", "a", "m"],
            },
            index=["q", "w", "e", "r", "t", "y", "u"],
        ),
        pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]}),
        pd.DataFrame(
            {
                "a": [None, None, np.nan, None],
                "b": [np.nan, None, np.nan, None],
            }
        ),
        pd.DataFrame({"a": []}),
        pd.DataFrame({"a": [np.nan], "b": [None]}),
        pd.DataFrame({"a": ["a", "b", "c", None, "e"]}),
        pd.DataFrame({"a": ["a", "b", "c", "d", "e"]}),
    ],
)
@pytest.mark.parametrize("api_call", ["isnull", "isna", "notna", "notnull"])
def test_dataframe_isnull_isna_and_reverse(na_data, nan_as_null, api_call):
    def detect_nan(x):
        # Check if the input is a float and if it is nan
        return x.apply(lambda v: isinstance(v, float) and np.isnan(v))

    df = na_data
    nan_contains = df.select_dtypes(object).apply(detect_nan)
    if nan_as_null is False and (
        nan_contains.any().any() and not nan_contains.all().all()
    ):
        with pytest.raises(cudf.errors.MixedTypeError):
            cudf.DataFrame(df, nan_as_null=nan_as_null)
    else:
        gdf = cudf.DataFrame(df, nan_as_null=nan_as_null)

        assert_eq(getattr(df, api_call)(), getattr(gdf, api_call)())

        # Test individual columns
        for col in df:
            assert_eq(
                getattr(df[col], api_call)(), getattr(gdf[col], api_call)()
            )
