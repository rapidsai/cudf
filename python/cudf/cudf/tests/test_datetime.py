import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

import cudf
from cudf.dataframe import DataFrame, Series
from cudf.dataframe.index import DatetimeIndex
from cudf.tests.utils import assert_eq


def data1():
    return pd.date_range("20010101", "20020215", freq="400h", name="times")


def data2():
    return pd.date_range(
        "20010101", "20020215", freq="400h", tz="US/Eastern", name="times"
    )


def numerical_data():
    return np.arange(1, 10)


fields = ["year", "month", "day", "hour", "minute", "second"]


@pytest.mark.parametrize("data", [data1(), data2()])
def test_series(data):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    np.testing.assert_equal(np.array(pd_data), np.array(gdf_data))


@pytest.mark.parametrize("data", [data1(), data2()])
def test_dt_ops(data):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(data.copy())

    assert_eq(pd_data == pd_data, gdf_data == gdf_data)
    assert_eq(pd_data < pd_data, gdf_data < gdf_data)
    assert_eq(pd_data > pd_data, gdf_data > gdf_data)


# libgdf doesn't respect timezones
@pytest.mark.parametrize("data", [data1()])
@pytest.mark.parametrize("field", fields)
def test_dt_series(data, field):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    base = getattr(pd_data.dt, field)
    test = getattr(gdf_data.dt, field).to_pandas().astype("int64")
    assert_series_equal(base, test)


@pytest.mark.parametrize("data", [data1()])
@pytest.mark.parametrize("field", fields)
def test_dt_index(data, field):
    pd_data = data.copy()
    gdf_data = DatetimeIndex(pd_data)
    assert_index_equal(
        getattr(gdf_data, field).to_pandas(), getattr(pd_data, field)
    )


def test_setitem_datetime():
    df = DataFrame()
    df["date"] = pd.date_range("20010101", "20010105").values
    assert np.issubdtype(df.date.dtype, np.datetime64)


def test_sort_datetime():
    df = pd.DataFrame()
    df["date"] = np.array(
        [
            np.datetime64("2016-11-20"),
            np.datetime64("2020-11-20"),
            np.datetime64("2019-11-20"),
            np.datetime64("1918-11-20"),
            np.datetime64("2118-11-20"),
        ]
    )
    df["vals"] = np.random.sample(len(df["date"]))

    gdf = cudf.from_pandas(df)

    s_df = df.sort_values(by="date")
    s_gdf = gdf.sort_values(by="date")

    assert_eq(s_df, s_gdf)


def test_issue_165():
    df_pandas = pd.DataFrame()
    start_date = dt.datetime.strptime("2000-10-21", "%Y-%m-%d")
    data = [(start_date + dt.timedelta(days=x)) for x in range(6)]
    df_pandas["dates"] = data
    df_pandas["num"] = [1, 2, 3, 4, 5, 6]
    df_cudf = DataFrame.from_pandas(df_pandas)

    base = df_pandas.query("dates==@start_date")
    test = df_cudf.query("dates==@start_date")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_cudf.dates == start_date
    base_mask = df_pandas.dates == start_date
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_ts = pd.Timestamp(start_date)
    test = df_cudf.query("dates==@start_date_ts")
    base = df_pandas.query("dates==@start_date_ts")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_cudf.dates == start_date_ts
    base_mask = df_pandas.dates == start_date_ts
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_np = np.datetime64(start_date_ts, "ns")
    test = df_cudf.query("dates==@start_date_np")
    base = df_pandas.query("dates==@start_date_np")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_cudf.dates == start_date_np
    base_mask = df_pandas.dates == start_date_np
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0


@pytest.mark.parametrize("data", [data1(), data2()])
@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
def test_typecast_from_datetime(data, dtype):
    pd_data = pd.Series(data.copy())
    np_data = np.array(pd_data).astype("datetime64[ms]")
    gdf_data = Series(pd_data)

    np_casted = np_data.astype(dtype)
    gdf_casted = gdf_data.astype(dtype)

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
def test_typecast_to_datetime(data, dtype):
    np_data = data.astype(dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype("datetime64[ms]")
    gdf_casted = gdf_data.astype("datetime64[ms]")

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize("nulls", ["some", "all"])
def test_to_from_pandas_nulls(data, nulls):
    pd_data = pd.Series(data.copy().astype("datetime64[ms]"))
    if nulls == "some":
        # Fill half the values with NaT
        pd_data[list(range(0, len(pd_data), 2))] = np.datetime64("nat")
    elif nulls == "all":
        # Fill all the values with NaT
        pd_data[:] = np.datetime64("nat")
    gdf_data = Series.from_pandas(pd_data)

    expect = pd_data
    got = gdf_data.to_pandas()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pytest.param(
            pd.Series([None, None], dtype="datetime64[ns]"),
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.parametrize(
    "nulls", ["none", pytest.param("some", marks=pytest.mark.xfail)]
)
def test_datetime_unique(data, nulls):
    psr = pd.Series(data)

    print(data)
    print(nulls)

    if len(data) > 0:
        if nulls == "some":
            p = np.random.randint(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.unique()
    got = gsr.unique()

    # convert to int64 for equivalence testing
    np.testing.assert_array_almost_equal(
        got.to_pandas().astype(int), expected.astype(int)
    )


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_nunique(data, nulls):
    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == "some":
            p = np.random.randint(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.nunique()
    got = gsr.nunique()
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_value_counts(data, nulls):
    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == "one":
            p = np.random.randint(0, len(data))
            psr[p] = None
        elif nulls == "some":
            p = np.random.randint(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.value_counts()
    got = gsr.value_counts()

    pandas_dict = expected.to_dict()
    gdf_dict = got.to_pandas().to_dict()

    assert pandas_dict == gdf_dict
