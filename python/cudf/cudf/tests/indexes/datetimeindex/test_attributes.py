# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "field",
    [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "weekday",
        "dayofweek",
        "dayofyear",
        "day_of_year",
    ],
)
def test_dt_index(field):
    data = pd.DatetimeIndex(
        [pd.Timestamp(2020, 1, 2, 3, 4, 5, 6, nanosecond=7)]
    )
    gdf_data = cudf.DatetimeIndex(data)
    assert_eq(getattr(gdf_data, field), getattr(data, field), exact=False)


@pytest.mark.parametrize(
    "attr",
    [
        "is_month_start",
        "is_month_end",
        "is_quarter_end",
        "is_quarter_start",
        "is_year_end",
        "is_year_start",
        "days_in_month",
        "timetz",
        "time",
        "date",
    ],
)
def test_dti_datetime_attributes(attr):
    data = [
        "2020-01-01",
        "2020-01-31",
        "2020-03-01",
        "2020-03-31",
        "2020-03-31",
        "2020-12-31",
        None,
    ]
    pd_dti = pd.DatetimeIndex(data, name="foo")
    cudf_dti = cudf.from_pandas(pd_dti)

    result = getattr(cudf_dti, attr)
    expected = getattr(pd_dti, attr)
    if isinstance(result, np.ndarray):
        # numpy doesn't assert object arrays with NaT correctly
        tm.assert_numpy_array_equal(result, expected)
    else:
        assert_eq(result, expected)


@pytest.mark.parametrize("attr", ["freq", "unit"])
def test_dti_properties(attr):
    pd_dti = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"
    )
    cudf_dti = cudf.DatetimeIndex(
        ["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"
    )

    result = getattr(cudf_dti, attr)
    expected = getattr(pd_dti, attr)
    assert result == expected


def test_writable_numpy_array():
    gi = cudf.Index([1, 2, 3], dtype="datetime64[ns]")
    expected_flags = pd.Index(
        [1, 2, 3], dtype="datetime64[ns]"
    )._data._ndarray.flags

    actual_flags = gi.to_pandas()._data._ndarray.flags
    assert expected_flags.c_contiguous == actual_flags.c_contiguous
    assert expected_flags.f_contiguous == actual_flags.f_contiguous
    assert expected_flags.writeable == actual_flags.writeable
    assert expected_flags.aligned == actual_flags.aligned
    assert expected_flags.writebackifcopy == actual_flags.writebackifcopy
