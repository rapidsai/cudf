# Copyright (c) 2019-2021, NVIDIA CORPORATION.

import datetime
import datetime as dt
import operator
import re

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core import DataFrame, Series
from cudf.core.index import DatetimeIndex
from cudf.testing._utils import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_eq,
    assert_exceptions_equal,
)


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


def timestamp_ms_data():
    return pd.Series(
        [
            "2019-07-16 00:00:00.333",
            "2019-07-16 00:00:00.666",
            "2019-07-16 00:00:00.888",
        ]
    )


def timestamp_us_data():
    return pd.Series(
        [
            "2019-07-16 00:00:00.333333",
            "2019-07-16 00:00:00.666666",
            "2019-07-16 00:00:00.888888",
        ]
    )


def timestamp_ns_data():
    return pd.Series(
        [
            "2019-07-16 00:00:00.333333333",
            "2019-07-16 00:00:00.666666666",
            "2019-07-16 00:00:00.888888888",
        ]
    )


def numerical_data():
    return np.arange(1, 10)


fields = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "weekday",
    "dayofweek",
    "dayofyear",
    "day_of_year",
]


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
    np.testing.assert_equal(np_data_1, gdf_data_1.to_array())
    np.testing.assert_equal(np_data_2, gdf_data_2.to_array())
    np.testing.assert_equal(
        np.less(np_data_1, np_data_2), (gdf_data_1 < gdf_data_2).to_array()
    )
    np.testing.assert_equal(
        np.greater(np_data_1, np_data_2), (gdf_data_1 > gdf_data_2).to_array()
    )
    np.testing.assert_equal(
        np.equal(np_data_1, np_data_2), (gdf_data_1 == gdf_data_2).to_array()
    )
    np.testing.assert_equal(
        np.less_equal(np_data_1, np_data_2),
        (gdf_data_1 <= gdf_data_2).to_array(),
    )
    np.testing.assert_equal(
        np.greater_equal(np_data_1, np_data_2),
        (gdf_data_1 >= gdf_data_2).to_array(),
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
    assert_eq(base, test)


@pytest.mark.parametrize("data", [data1()])
@pytest.mark.parametrize("field", fields)
def test_dt_index(data, field):
    pd_data = data.copy()
    gdf_data = DatetimeIndex(pd_data)
    assert_eq(getattr(gdf_data, field), getattr(pd_data, field))


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
    assert_eq(base, test)
    assert len(test) > 0

    mask = df_cudf.dates == start_date
    base_mask = df_pandas.dates == start_date
    assert_eq(mask, base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_ts = pd.Timestamp(start_date)
    test = df_cudf.query("dates==@start_date_ts")
    base = df_pandas.query("dates==@start_date_ts")
    assert_eq(base, test)
    assert len(test) > 0

    mask = df_cudf.dates == start_date_ts
    base_mask = df_pandas.dates == start_date_ts
    assert_eq(mask, base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_np = np.datetime64(start_date_ts, "ns")
    test = df_cudf.query("dates==@start_date_np")
    base = df_pandas.query("dates==@start_date_np")
    assert_eq(base, test)
    assert len(test) > 0

    mask = df_cudf.dates == start_date_np
    base_mask = df_pandas.dates == start_date_np
    assert_eq(mask, base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0


@pytest.mark.parametrize("data", [data1(), data2()])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_typecast_from_datetime(data, dtype):
    pd_data = pd.Series(data.copy())
    np_data = np.array(pd_data)
    gdf_data = Series(pd_data)

    np_casted = np_data.astype(dtype)
    gdf_casted = gdf_data.astype(dtype)

    np.testing.assert_equal(np_casted, gdf_casted.to_array())


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

    np.testing.assert_equal(np_casted, gdf_casted.to_array())


@pytest.mark.parametrize("data", [timeseries_us_data()])
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_different_datetime_resolutions(data, dtype):
    pd_data = pd.Series(data.copy())
    np_data = np.array(pd_data).astype(dtype)
    gdf_series = Series(pd_data).astype(dtype)
    np.testing.assert_equal(np_data, gdf_series.to_array())


@pytest.mark.parametrize(
    "data", [timestamp_ms_data(), timestamp_us_data(), timestamp_ns_data()]
)
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_string_timstamp_typecast_to_different_datetime_resolutions(
    data, dtype
):
    pd_sr = data
    gdf_sr = cudf.Series.from_pandas(pd_sr)

    expect = pd_sr.values.astype(dtype)
    got = gdf_sr.astype(dtype).values_host

    np.testing.assert_equal(expect, got)


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize("from_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_datetime(data, from_dtype, to_dtype):
    np_data = data.astype(from_dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype(to_dtype)
    gdf_casted = gdf_data.astype(to_dtype)

    np.testing.assert_equal(np_casted, gdf_casted.to_array())


@pytest.mark.parametrize("data", [numerical_data()])
@pytest.mark.parametrize("from_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_from_datetime(data, from_dtype, to_dtype):
    np_data = data.astype(from_dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype(to_dtype).astype(from_dtype)
    gdf_casted = gdf_data.astype(to_dtype).astype(from_dtype)

    np.testing.assert_equal(np_casted, gdf_casted.to_array())


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
    pd_data = pd.Series(data.copy().astype("datetime64[ns]"))
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
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize(
    "nulls", ["none", pytest.param("some", marks=pytest.mark.xfail)]
)
def test_datetime_unique(data, nulls):
    psr = data.copy()

    if len(data) > 0:
        if nulls == "some":
            p = np.random.randint(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.unique()
    got = gsr.unique()

    assert_eq(pd.Series(expected), got.to_pandas())


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_nunique(data, nulls):
    psr = data.copy()

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
            np.array(
                ["2018-01-01", None, "2019-12-30"], dtype="datetime64[ms]"
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


def test_datetime_dataframe():
    data = {
        "timearray": np.array(
            [0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]"
        )
    }
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    assert_eq(pdf, gdf)

    assert_eq(pdf.dropna(), gdf.dropna())

    assert_eq(pdf.isnull(), gdf.isnull())

    data = np.array([0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]")
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(ps, gs)

    assert_eq(ps.dropna(), gs.dropna())

    assert_eq(ps.isnull(), gs.isnull())


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        pd.Series([], dtype="float64"),
        pd.Index([]),
        pd.Series([1, 2, 3]),
        pd.Series([0, 1, -1]),
        pd.Series([0, 1, -1, 100.3, 200, 47637289]),
        pd.Series(["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"]),
        [1, 2, 3, 100, -123, -1, 0, 1000000000000679367],
        pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}),
        pd.DataFrame(
            {"year": ["2015", "2016"], "month": ["2", "3"], "day": [4, 5]}
        ),
        pd.DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "minute": [1, 100],
                "second": [90, 10],
                "hour": [1, 0.5],
            },
            index=["a", "b"],
        ),
        pd.DataFrame(
            {
                "year": [],
                "month": [],
                "day": [],
                "minute": [],
                "second": [],
                "hour": [],
            },
        ),
        ["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"],
        pd.Index([1, 2, 3, 4]),
        pd.DatetimeIndex(
            ["1970-01-01 00:00:00.000000001", "1970-01-01 00:00:00.000000002"],
            dtype="datetime64[ns]",
            freq=None,
        ),
        pd.DatetimeIndex([], dtype="datetime64[ns]", freq=None,),
        pd.Series([1, 2, 3]).astype("datetime64[ns]"),
        pd.Series([1, 2, 3]).astype("datetime64[us]"),
        pd.Series([1, 2, 3]).astype("datetime64[ms]"),
        pd.Series([1, 2, 3]).astype("datetime64[s]"),
        pd.Series([1, 2, 3]).astype("datetime64[D]"),
        1,
        100,
        17,
        53.638435454,
        np.array([1, 10, 15, 478925, 2327623467]),
        np.array([0.3474673, -10, 15, 478925.34345, 2327623467]),
    ],
)
@pytest.mark.parametrize("dayfirst", [True, False])
@pytest.mark.parametrize("infer_datetime_format", [True, False])
def test_cudf_to_datetime(data, dayfirst, infer_datetime_format):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        if type(pd_data).__module__ == np.__name__:
            gd_data = cp.array(pd_data)
        else:
            gd_data = pd_data

    expected = pd.to_datetime(
        pd_data, dayfirst=dayfirst, infer_datetime_format=infer_datetime_format
    )
    actual = cudf.to_datetime(
        gd_data, dayfirst=dayfirst, infer_datetime_format=infer_datetime_format
    )

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        "2",
        ["1", "2", "3"],
        ["1/1/1", "2/2/2", "1"],
        pd.Series([1, 2, 3], dtype="timedelta64[ns]"),
        pd.DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "minute": [1, 100],
                "second": [90, 10],
                "hour": [1, 0],
                "blablacol": [1, 1],
            }
        ),
        pd.DataFrame(
            {
                "month": [2, 3],
                "day": [4, 5],
                "minute": [1, 100],
                "second": [90, 10],
                "hour": [1, 0],
            }
        ),
    ],
)
def test_to_datetime_errors(data):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        gd_data = pd_data

    assert_exceptions_equal(
        pd.to_datetime, cudf.to_datetime, ([pd_data],), ([gd_data],)
    )


def test_to_datetime_not_implemented():

    with pytest.raises(NotImplementedError):
        cudf.to_datetime([], exact=False)

    with pytest.raises(NotImplementedError):
        cudf.to_datetime([], origin="julian")

    with pytest.raises(NotImplementedError):
        cudf.to_datetime([], yearfirst=True)


@pytest.mark.parametrize(
    "data",
    [
        1,
        [],
        pd.Series([], dtype="float64"),
        pd.Index([]),
        pd.Series([1, 2, 3]),
        pd.Series([1, 2.4, 3]),
        pd.Series([0, 1, -1]),
        pd.Series([0, 1, -1, 100, 200, 47637]),
        [10, 12, 1200, 15003],
        pd.DatetimeIndex([], dtype="datetime64[ns]", freq=None,),
        pd.Index([1, 2, 3, 4]),
    ],
)
@pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
def test_to_datetime_units(data, unit):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        gd_data = pd_data

    expected = pd.to_datetime(pd_data, unit=unit)
    actual = cudf.to_datetime(gd_data, unit=unit)

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data,format",
    [
        ("2012-10-11", None),
        ("2012-10-11", "%Y-%m-%d"),
        ("2012-10-11", "%Y-%d-%m"),
        (["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"], None),
        (["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"], "%Y-%m-%d"),
        (["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"], "%Y-%d-%m"),
        (["10-11-2012", "01-01-2010", "07-07-2016", "02-02-2014"], "%m-%d-%Y"),
        (["10-11-2012", "01-01-2010", "07-07-2016", "02-02-2014"], "%d-%m-%Y"),
        (["10-11-2012", "01-01-2010", "07-07-2016", "02-02-2014"], None),
        (["2012/10/11", "2010/01/01", "2016/07/07", "2014/02/02"], None),
        (["2012/10/11", "2010/01/01", "2016/07/07", "2014/02/02"], "%Y/%m/%d"),
        (["2012/10/11", "2010/01/01", "2016/07/07", "2014/02/02"], "%Y/%d/%m"),
        (["10/11/2012", "01/01/2010", "07/07/2016", "02/02/2014"], "%m/%d/%Y"),
        (["10/11/2012", "01/01/2010", "07/07/2016", "02/02/2014"], "%d/%m/%Y"),
        (["10/11/2012", "01/01/2010", "07/07/2016", "02/02/2014"], None),
        (pd.Series([2015, 2020, 2021]), "%Y"),
        pytest.param(
            pd.Series(["1", "2", "1"]),
            "%m",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6109"
                "https://github.com/pandas-dev/pandas/issues/35934"
            ),
        ),
        pytest.param(
            pd.Series(["14", "20", "10"]),
            "%d",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6109"
                "https://github.com/pandas-dev/pandas/issues/35934"
            ),
        ),
        (pd.Series([2015, 2020.0, 2021.2]), "%Y"),
    ],
)
@pytest.mark.parametrize("infer_datetime_format", [True, False])
def test_to_datetime_format(data, format, infer_datetime_format):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        gd_data = pd_data

    expected = pd.to_datetime(
        pd_data, format=format, infer_datetime_format=infer_datetime_format
    )
    actual = cudf.to_datetime(
        gd_data, format=format, infer_datetime_format=infer_datetime_format
    )

    assert_eq(actual, expected)


def test_datetime_can_cast_safely():

    sr = cudf.Series(
        ["1679-01-01", "2000-01-31", "2261-01-01"], dtype="datetime64[ms]"
    )
    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]"))

    sr = cudf.Series(
        ["1677-01-01", "2000-01-31", "2263-01-01"], dtype="datetime64[ms]"
    )

    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]")) is False


# Cudf autocasts unsupported time_units
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[D]", "datetime64[W]", "datetime64[M]", "datetime64[Y]"],
)
def test_datetime_array_timeunit_cast(dtype):
    testdata = np.array(
        [
            np.datetime64("2016-11-20"),
            np.datetime64("2020-11-20"),
            np.datetime64("2019-11-20"),
            np.datetime64("1918-11-20"),
            np.datetime64("2118-11-20"),
        ],
        dtype=dtype,
    )

    gs = Series(testdata)
    ps = pd.Series(testdata)

    assert_eq(ps, gs)

    gdf = DataFrame()
    gdf["a"] = np.arange(5)
    gdf["b"] = testdata

    pdf = pd.DataFrame()
    pdf["a"] = np.arange(5)
    pdf["b"] = testdata
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("timeunit", ["D", "W", "M", "Y"])
def test_datetime_scalar_timeunit_cast(timeunit):
    testscalar = np.datetime64("2016-11-20", timeunit)

    gs = Series(testscalar)
    ps = pd.Series(testscalar)
    assert_eq(ps, gs)

    gdf = DataFrame()
    gdf["a"] = np.arange(5)
    gdf["b"] = testscalar

    pdf = pd.DataFrame()
    pdf["a"] = np.arange(5)
    pdf["b"] = testscalar

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        ["2001-01-01", "2002-02-02", "2000-01-05", "NaT"],
        ["2001-01-01", "2002-02-02", "2000-01-05", None],
        [None, None, None, None, None],
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_str_null_to_datetime(data, dtype):
    psr = pd.Series(data)
    gsr = Series(data)

    assert_eq(psr.astype(dtype), gsr.astype(dtype))


def test_str_to_datetime_error():
    psr = pd.Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])
    gsr = Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=(["datetime64[s]"],),
        rfunc_args_and_kwargs=(["datetime64[s]"],),
        check_exception_type=False,
        expected_error_message=re.escape(
            "Could not convert `None` value to datetime"
        ),
    )


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
        np.datetime64("2005-02"),
        np.datetime64("2005-02-25"),
        np.datetime64("2005-02-25T03:30"),
        np.datetime64("nat"),
    ],
)
@pytest.mark.parametrize("data_dtype", DATETIME_TYPES)
@pytest.mark.parametrize("other_dtype", DATETIME_TYPES)
def test_datetime_subtract(data, other, data_dtype, other_dtype):

    gsr = cudf.Series(data, dtype=data_dtype)
    psr = gsr.to_pandas()

    if isinstance(other, np.datetime64):
        gsr_other = other
        psr_other = other
    else:
        gsr_other = cudf.Series(other, dtype=other_dtype)
        psr_other = gsr_other.to_pandas()

    expected = psr - psr_other
    actual = gsr - gsr_other

    assert_eq(expected, actual)

    expected = psr_other - psr
    actual = gsr_other - gsr

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "other_scalars",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=447),
        datetime.timedelta(hours=447),
        datetime.timedelta(weeks=734),
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize(
    "op", ["add", "sub"],
)
def test_datetime_series_ops_with_scalars(data, other_scalars, dtype, op):
    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas()

    if op == "add":
        expected = psr + other_scalars
        actual = gsr + other_scalars
    elif op == "sub":
        expected = psr - other_scalars
        actual = gsr - other_scalars

    assert_eq(expected, actual)

    if op == "add":
        expected = other_scalars + psr
        actual = other_scalars + gsr

        assert_eq(expected, actual)

    elif op == "sub":
        assert_exceptions_equal(
            lfunc=operator.sub,
            rfunc=operator.sub,
            lfunc_args_and_kwargs=([other_scalars, psr],),
            rfunc_args_and_kwargs=([other_scalars, gsr],),
            compare_error_message=False,
        )


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        pytest.param(np.timedelta64("nat"), marks=pytest.mark.xfail),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize("op", [np.add, np.subtract])
def test_datetime_series_ops_with_cudf_scalars(data, scalar, dtype, op):
    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas()

    expect = op(psr, scalar)
    got = op(gsr, cudf.Scalar(scalar))

    assert_eq(expect, got)


def test_datetime_invalid_ops():
    sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
        compare_error_message=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3],
        [None, 1, 10, 11, None],
        [None, None, None, None, None],
        [None],
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize(
    "fill_value",
    [
        np.datetime64("2005-02"),
        np.datetime64("2005-02-25"),
        np.datetime64("2005-02-25T03:30"),
        np.datetime64("nat"),
    ],
)
def test_datetime_fillna(data, dtype, fill_value):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    expected = psr.dropna()
    actual = sr.dropna()

    assert_eq(expected, actual)

    expected = psr.fillna(fill_value)
    actual = sr.fillna(fill_value)

    assert_eq(expected, actual)

    expected = expected.dropna()
    actual = actual.dropna()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data", [[1, 2, 3, None], [], [100121, 1221312, 321312321, 1232131223]]
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize(
    "date_format", ["%d - %m", "%y/%H", "%Y", "%I - %M / %S", "%f", "%j", "%p"]
)
def test_datetime_strftime(data, dtype, date_format):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

    expected = psr.dt.strftime(date_format=date_format)
    actual = gsr.dt.strftime(date_format=date_format)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "date_format",
    [
        "%a",
        "%A",
        "%w",
        "%b",
        "%B",
        "%U",
        "%W",
        "%c",
        "%x",
        "%X",
        "%G",
        "%u",
        "%V",
    ],
)
def test_datetime_strftime_not_implemented_formats(date_format):
    gsr = cudf.Series([1, 2, 3], dtype="datetime64[ms]")

    with pytest.raises(NotImplementedError):
        gsr.dt.strftime(date_format=date_format)


@pytest.mark.parametrize("data", [[1, 2, 3], [], [1, 20, 1000, None]])
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize("stat", ["mean", "quantile"])
def test_datetime_stats(data, dtype, stat):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

    expected = getattr(psr, stat)()
    actual = getattr(gsr, stat)()

    if len(data) == 0:
        assert np.isnat(expected.to_numpy()) and np.isnat(actual.to_numpy())
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize("op", ["max", "min", "std", "median"])
@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 100],
        [10, None, 100, None, None],
        [None, None, None],
        [1231],
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_datetime_reductions(data, op, dtype):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    actual = getattr(sr, op)()
    expected = getattr(psr, op)()

    if (
        expected is pd.NaT
        and actual is pd.NaT
        or (np.isnat(expected.to_numpy()) and np.isnat(actual))
    ):
        assert True
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        np.datetime_as_string(
            np.arange("2002-10-27T04:30", 4 * 60, 60, dtype="M8[m]"),
            timezone="UTC",
        ),
        np.datetime_as_string(
            np.arange("2002-10-27T04:30", 10 * 60, 1, dtype="M8[m]"),
            timezone="UTC",
        ),
        np.datetime_as_string(
            np.arange("2002-10-27T04:30", 10 * 60, 1, dtype="M8[ns]"),
            timezone="UTC",
        ),
        np.datetime_as_string(
            np.arange("2002-10-27T04:30", 10 * 60, 1, dtype="M8[us]"),
            timezone="UTC",
        ),
        np.datetime_as_string(
            np.arange("2002-10-27T04:30", 4 * 60, 60, dtype="M8[s]"),
            timezone="UTC",
        ),
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_datetime_infer_format(data, dtype):
    sr = cudf.Series(data)
    psr = pd.Series(data)

    expected = psr.astype(dtype)
    actual = sr.astype(dtype)

    assert_eq(expected, actual)


def test_dateoffset_instance_subclass_check():
    assert not issubclass(pd.DateOffset, cudf.DateOffset)
    assert not isinstance(pd.DateOffset(), cudf.DateOffset)


def test_datetime_to_datetime_error():
    assert_exceptions_equal(
        lfunc=pd.to_datetime,
        rfunc=cudf.to_datetime,
        lfunc_args_and_kwargs=(["02-Oct-2017 09:30", "%d-%B-%Y %H:%M"],),
        rfunc_args_and_kwargs=(["02-Oct-2017 09:30", "%d-%B-%Y %H:%M"],),
        check_exception_type=False,
        expected_error_message=re.escape(
            "errors parameter has to be either one of: ['ignore', 'raise', "
            "'coerce', 'warn'], found: %d-%B-%Y %H:%M"
        ),
    )
