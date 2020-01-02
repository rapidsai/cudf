import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.util.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

import rmm

import cudf
from cudf.core import DataFrame, Series
from cudf.core.index import DatetimeIndex
from cudf.tests.utils import assert_eq


def data1():
    return pd.date_range("20010101", "20020215", freq="400h", name="times")


def data2():
    return pd.date_range("20010101", "20020215", freq="400h", name="times")


def timeseries_us_data():
    return pd.date_range(
        "2019-07-16 00:00:00",
        "2019-07-16 00:00:01",
        freq="5555us",
        name="times",
    )


def numerical_data():
    return np.arange(1, 10)


fields = ["year", "month", "day", "hour", "minute", "second", "weekday"]


@pytest.mark.parametrize("data", [data1(), data2()])
def test_series(data):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    assert_eq(pd_data, gdf_data)


@pytest.mark.parametrize(
    "lhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "rhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_datetime_series_binops_pandas(lhs_dtype, rhs_dtype):
    pd_data_1 = pd.Series(
        pd.date_range("20010101", "20020215", freq="400h", name="times")
    )
    pd_data_2 = pd.Series(
        pd.date_range("20010101", "20020215", freq="401h", name="times")
    )
    gdf_data_1 = Series(pd_data_1).astype(lhs_dtype)
    gdf_data_2 = Series(pd_data_2).astype(rhs_dtype)
    assert_eq(pd_data_1, gdf_data_1.astype("datetime64[ns]"))
    assert_eq(pd_data_2, gdf_data_2.astype("datetime64[ns]"))
    assert_eq(pd_data_1 < pd_data_2, gdf_data_1 < gdf_data_2)
    assert_eq(pd_data_1 > pd_data_2, gdf_data_1 > gdf_data_2)
    assert_eq(pd_data_1 == pd_data_2, gdf_data_1 == gdf_data_2)
    assert_eq(pd_data_1 <= pd_data_2, gdf_data_1 <= gdf_data_2)
    assert_eq(pd_data_1 >= pd_data_2, gdf_data_1 >= gdf_data_2)


@pytest.mark.parametrize(
    "lhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "rhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_datetime_series_binops_numpy(lhs_dtype, rhs_dtype):
    pd_data_1 = pd.Series(
        pd.date_range("20010101", "20020215", freq="400h", name="times")
    )
    pd_data_2 = pd.Series(
        pd.date_range("20010101", "20020215", freq="401h", name="times")
    )
    gdf_data_1 = Series(pd_data_1).astype(lhs_dtype)
    gdf_data_2 = Series(pd_data_2).astype(rhs_dtype)
    np_data_1 = np.array(pd_data_1).astype(lhs_dtype)
    np_data_2 = np.array(pd_data_2).astype(rhs_dtype)
    np.testing.assert_equal(np_data_1, np.array(gdf_data_1))
    np.testing.assert_equal(np_data_2, np.array(gdf_data_2))
    np.testing.assert_equal(
        np.less(np_data_1, np_data_2), gdf_data_1 < gdf_data_2
    )
    np.testing.assert_equal(
        np.greater(np_data_1, np_data_2), gdf_data_1 > gdf_data_2
    )
    np.testing.assert_equal(
        np.equal(np_data_1, np_data_2), gdf_data_1 == gdf_data_2
    )
    np.testing.assert_equal(
        np.less_equal(np_data_1, np_data_2), gdf_data_1 <= gdf_data_2
    )
    np.testing.assert_equal(
        np.greater_equal(np_data_1, np_data_2), gdf_data_1 >= gdf_data_2
    )


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
    np_data = np.array(pd_data)
    gdf_data = Series(pd_data)

    np_casted = np_data.astype(dtype)
    gdf_casted = gdf_data.astype(dtype)

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize("data", [data1(), data2()])
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_from_datetime_to_int64_to_datetime(data, dtype):
    pd_data = pd.Series(data.copy())
    np_data = np.array(pd_data)
    gdf_data = Series(pd_data)

    np_casted = np_data.astype(np.int64).astype(dtype)
    gdf_casted = gdf_data.astype(np.int64).astype(dtype)

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize("data", [timeseries_us_data()])
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_different_datetime_resolutions(data, dtype):
    pd_data = pd.Series(data.copy())
    np_data = np.array(pd_data).astype(dtype)
    gdf_series = Series(pd_data).astype(dtype)
    np.testing.assert_equal(np_data, np.array(gdf_series))


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize(
    "from_dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_datetime(data, from_dtype, to_dtype):
    np_data = data.astype(from_dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype(to_dtype)
    gdf_casted = gdf_data.astype(to_dtype)

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize(
    "from_dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_from_datetime(data, from_dtype, to_dtype):
    np_data = data.astype(from_dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype(to_dtype).astype(from_dtype)
    gdf_casted = gdf_data.astype(to_dtype).astype(from_dtype)

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize(
    "from_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_from_datetime_to_datetime(data, from_dtype, to_dtype):
    np_data = data.astype(from_dtype)
    gdf_col = Series(np_data)._column

    np_casted = np_data.astype(to_dtype)
    gdf_casted = gdf_col.astype(to_dtype)

    np.testing.assert_equal(np_casted, gdf_casted.to_array())


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize("nulls", ["some", "all"])
def test_to_from_pandas_nulls(data, nulls):
    pd_data = pd.Series(data.copy())
    if nulls == "some":
        # Fill half the values with NaT
        pd_data[list(range(0, len(pd_data), 2))] = np.datetime64("nat", "ns")
    elif nulls == "all":
        # Fill all the values with NaT
        pd_data[:] = np.datetime64("nat", "ns")
    gdf_data = Series.from_pandas(pd_data)

    expect = pd_data
    got = gdf_data.to_pandas()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_datetime_to_arrow(dtype):
    timestamp = (
        cudf.datasets.timeseries(
            start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={}
        )
        .reset_index()["timestamp"]
        .reset_index(drop=True)
    )
    gdf = DataFrame({"timestamp": timestamp.astype(dtype)})
    assert_eq(gdf, DataFrame.from_arrow(gdf.to_arrow(preserve_index=False)))


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


testdata = [
    (
        Series(
            ["2018-01-01", None, "2019-01-31", None, "2018-01-01"],
            dtype="datetime64[ms]",
        ),
        True,
    ),
    (
        Series(
            [
                "2018-01-01",
                "2018-01-02",
                "2019-01-31",
                "2018-03-01",
                "2018-01-01",
            ],
            dtype="datetime64[ms]",
        ),
        False,
    ),
    (
        Series(
            rmm.to_device(
                np.array(
                    ["2018-01-01", None, "2019-12-30"], dtype="datetime64[ms]"
                )
            )
        ),
        True,
    ),
]


@pytest.mark.parametrize("data, expected", testdata)
def test_datetime_has_null_test(data, expected):
    pd_data = data.to_pandas()
    count = pd_data.notna().value_counts()
    expected_count = 0
    if False in count.keys():
        expected_count = count[False]

    assert_eq(expected, data.has_nulls)
    assert_eq(expected_count, data.null_count)


def test_datetime_has_null_test_pyarrow():
    data = Series(
        pa.array(
            [0, np.iinfo("int64").min, np.iinfo("int64").max, None],
            type=pa.timestamp("ns"),
        )
    )
    expected = True
    expected_count = 1

    assert_eq(expected, data.has_nulls)
    assert_eq(expected_count, data.null_count)
