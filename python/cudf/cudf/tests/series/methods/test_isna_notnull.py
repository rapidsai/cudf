# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.errors import MixedTypeError
from cudf.testing import assert_eq


@pytest.fixture(
    params=[
        pd.Series([0, 1, 2, np.nan, 4, None, 6]),
        pd.Series(
            [0, 1, 2, np.nan, 4, None, 6],
            index=["q", "w", "e", "r", "t", "y", "u"],
            name="a",
        ),
        pd.Series([0, 1, 2, 3, 4]),
        pd.Series(["a", "b", "u", "h", "d"]),
        pd.Series([None, None, np.nan, None, np.inf, -np.inf]),
        pd.Series([], dtype="float64"),
        pd.Series(
            [pd.NaT, pd.Timestamp("1939-05-27"), pd.Timestamp("1940-04-25")]
        ),
        pd.Series([np.nan]),
        pd.Series([None]),
        pd.Series(["a", "b", "", "c", None, "e"]),
    ]
)
def ps(request):
    return request.param


def test_series_isnull_isna(ps, nan_as_null):
    nan_contains = ps.apply(lambda x: isinstance(x, float) and np.isnan(x))
    if nan_as_null is False and (
        nan_contains.any() and not nan_contains.all() and ps.dtype == object
    ):
        with pytest.raises(MixedTypeError):
            cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)
    else:
        gs = cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)

        assert_eq(ps.isnull(), gs.isnull())
        assert_eq(ps.isna(), gs.isna())


def test_series_notnull_notna(ps, nan_as_null):
    nan_contains = ps.apply(lambda x: isinstance(x, float) and np.isnan(x))
    if nan_as_null is False and (
        nan_contains.any() and not nan_contains.all() and ps.dtype == object
    ):
        with pytest.raises(MixedTypeError):
            cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)
    else:
        gs = cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)

        assert_eq(ps.notnull(), gs.notnull())
        assert_eq(ps.notna(), gs.notna())
