import math

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_110
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "data,index",
    [
        ([], []),
        ([1, 1, 1, 1], None),
        ([1, 2, 3, 4], pd.date_range("2001-01-01", "2001-01-04")),
        ([1, 2, 4, 9, 9, 4], ["a", "b", "c", "d", "e", "f"]),
    ],
)
@pytest.mark.parametrize("agg", ["sum", "min", "max", "mean", "count"])
@pytest.mark.parametrize("nulls", ["none", "one", "some", "all"])
@pytest.mark.parametrize("center", [True, False])
def test_rolling_series_basic(data, index, agg, nulls, center):
    if PANDAS_GE_110:
        kwargs = {"check_freq": False}
    else:
        kwargs = {}
    if len(data) > 0:
        if nulls == "one":
            p = np.random.randint(0, len(data))
            data[p] = np.nan
        elif nulls == "some":
            p1, p2 = np.random.randint(0, len(data), (2,))
            data[p1] = np.nan
            data[p2] = np.nan
        elif nulls == "all":
            data = [np.nan] * len(data)

    psr = pd.Series(data, index=index)
    gsr = cudf.Series(psr)
    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            expect = getattr(
                psr.rolling(window_size, min_periods, center), agg
            )().fillna(-1)
            got = getattr(
                gsr.rolling(window_size, min_periods, center), agg
            )().fillna(-1)
            assert_eq(expect, got, check_dtype=False, **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [], "b": []},
        {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]},
        {"a": [1, 2, 4, 9, 9, 4], "b": [1, 2, 4, 9, 9, 4]},
        {
            "a": np.array([1, 2, 4, 9, 9, 4]),
            "b": np.array([1.5, 2.2, 2.2, 8.0, 9.1, 4.2]),
        },
    ],
)
@pytest.mark.parametrize("agg", ["sum", "min", "max", "mean", "count"])
@pytest.mark.parametrize("nulls", ["none", "one", "some", "all"])
@pytest.mark.parametrize("center", [True, False])
def test_rolling_dataframe_basic(data, agg, nulls, center):
    pdf = pd.DataFrame(data)

    if len(pdf) > 0:
        for col_name in pdf.columns:
            if nulls == "one":
                p = np.random.randint(0, len(data))
                pdf[col_name][p] = np.nan
            elif nulls == "some":
                p1, p2 = np.random.randint(0, len(data), (2,))
                pdf[col_name][p1] = np.nan
                pdf[col_name][p2] = np.nan
            elif nulls == "all":
                pdf[col_name][:] = np.nan

    gdf = cudf.from_pandas(pdf)
    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            expect = getattr(
                pdf.rolling(window_size, min_periods, center), agg
            )().fillna(-1)
            got = getattr(
                gdf.rolling(window_size, min_periods, center), agg
            )().fillna(-1)
            assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "agg",
    [
        pytest.param("sum"),
        pytest.param("min"),
        pytest.param("max"),
        pytest.param("mean"),
        pytest.param(
            "count",  # Does not follow similar conventions as
            # with non-offset columns
            marks=pytest.mark.xfail(
                reason="Differs from pandas behaviour here"
            ),
        ),
    ],
)
def test_rolling_with_offset(agg):
    psr = pd.Series(
        [1, 2, 4, 4, np.nan, 9],
        index=[
            pd.Timestamp("20190101 09:00:00"),
            pd.Timestamp("20190101 09:00:01"),
            pd.Timestamp("20190101 09:00:02"),
            pd.Timestamp("20190101 09:00:04"),
            pd.Timestamp("20190101 09:00:07"),
            pd.Timestamp("20190101 09:00:08"),
        ],
    )
    gsr = cudf.from_pandas(psr)
    assert_eq(
        getattr(psr.rolling("2s"), agg)().fillna(-1),
        getattr(gsr.rolling("2s"), agg)().fillna(-1),
        check_dtype=False,
    )


def test_rolling_count_with_offset():
    """
    This test covers the xfail case from test_rolling_with_offset["count"].
    It is expected that count should return a non-Nan value, even if
    the counted value is a Nan, unless the min-periods condition
    is not met.
    This behaviour is consistent with counts for rolling-windows,
    in the non-offset window case.
    """
    psr = pd.Series(
        [1, 2, 4, 4, np.nan, 9],
        index=[
            pd.Timestamp("20190101 09:00:00"),
            pd.Timestamp("20190101 09:00:01"),
            pd.Timestamp("20190101 09:00:02"),
            pd.Timestamp("20190101 09:00:04"),
            pd.Timestamp("20190101 09:00:07"),
            pd.Timestamp("20190101 09:00:08"),
        ],
    )
    gsr = cudf.from_pandas(psr)
    assert_eq(
        getattr(gsr.rolling("2s"), "count")().fillna(-1),
        pd.Series(
            [1, 2, 2, 1, 0, 1],
            index=[
                pd.Timestamp("20190101 09:00:00"),
                pd.Timestamp("20190101 09:00:01"),
                pd.Timestamp("20190101 09:00:02"),
                pd.Timestamp("20190101 09:00:04"),
                pd.Timestamp("20190101 09:00:07"),
                pd.Timestamp("20190101 09:00:08"),
            ],
        ),
        check_dtype=False,
    )


def test_rolling_getattr():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.rolling(2).a.sum().fillna(-1),
        gdf.rolling(2).a.sum().fillna(-1),
        check_dtype=False,
    )


def test_rolling_getitem():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.rolling(2)["a"].sum().fillna(-1),
        gdf.rolling(2)["a"].sum().fillna(-1),
        check_dtype=False,
    )
    assert_eq(
        pdf.rolling(2)["a", "b"].sum().fillna(-1),
        gdf.rolling(2)["a", "b"].sum().fillna(-1),
        check_dtype=False,
    )
    assert_eq(
        pdf.rolling(2)[["a", "b"]].sum().fillna(-1),
        gdf.rolling(2)["a", "b"].sum().fillna(-1),
        check_dtype=False,
    )


def test_rolling_getitem_window():
    if PANDAS_GE_110:
        kwargs = {"check_freq": False}
    else:
        kwargs = {}
    index = pd.DatetimeIndex(
        pd.date_range("2000-01-01", "2000-01-02", freq="1h")
    )
    pdf = pd.DataFrame({"x": np.arange(len(index))}, index=index)
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.rolling("2h").x.mean(), gdf.rolling("2h").x.mean(), **kwargs)


@pytest.mark.parametrize(
    "data,index", [([1.2, 4.5, 5.9, 2.4, 9.3, 7.1], None), ([], [])]
)
@pytest.mark.parametrize("center", [True, False])
def test_rollling_series_numba_udf_basic(data, index, center):

    psr = pd.Series(data, index=index)
    gsr = cudf.from_pandas(psr)

    def some_func(A):
        b = 0
        for a in A:
            b = max(b, math.sqrt(a))
        return b

    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            assert_eq(
                psr.rolling(window_size, min_periods, center)
                .apply(some_func)
                .fillna(-1),
                gsr.rolling(window_size, min_periods, center)
                .apply(some_func)
                .fillna(-1),
                check_dtype=False,
            )


@pytest.mark.parametrize(
    "data",
    [
        {"a": [], "b": []},
        {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]},
        {"a": [1, 2, 4, 9, 9, 4], "b": [1, 2, 4, 9, 9, 4]},
        {
            "a": np.array([1, 2, 4, 9, 9, 4]),
            "b": np.array([1.5, 2.2, 2.2, 8.0, 9.1, 4.2]),
        },
    ],
)
@pytest.mark.parametrize("center", [True, False])
def test_rolling_dataframe_numba_udf_basic(data, center):

    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    def some_func(A):
        b = 0
        for a in A:
            b = b + a ** 2
        return b / len(A)

    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            assert_eq(
                pdf.rolling(window_size, min_periods, center)
                .apply(some_func)
                .fillna(-1),
                gdf.rolling(window_size, min_periods, center)
                .apply(some_func)
                .fillna(-1),
                check_dtype=False,
            )


def test_rolling_numba_udf_with_offset():
    psr = pd.Series(
        [1, 2, 4, 4, 8, 9],
        index=[
            pd.Timestamp("20190101 09:00:00"),
            pd.Timestamp("20190101 09:00:01"),
            pd.Timestamp("20190101 09:00:02"),
            pd.Timestamp("20190101 09:00:04"),
            pd.Timestamp("20190101 09:00:07"),
            pd.Timestamp("20190101 09:00:08"),
        ],
    )
    gsr = cudf.from_pandas(psr)

    def some_func(A):
        b = 0
        for a in A:
            b = b + a
        return b / len(A)

    assert_eq(
        psr.rolling("2s").apply(some_func).fillna(-1),
        gsr.rolling("2s").apply(some_func).fillna(-1),
        check_dtype=False,
    )


@pytest.mark.parametrize("agg", ["sum", "min", "max", "mean", "count"])
def test_rolling_groupby_simple(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        }
    )
    gdf = cudf.from_pandas(pdf)

    for window_size in range(1, len(pdf) + 1):
        expect = getattr(pdf.groupby("a").rolling(window_size), agg)().fillna(
            -1
        )
        got = getattr(gdf.groupby("a").rolling(window_size), agg)().fillna(-1)
        assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("agg", ["sum", "min", "max", "mean", "count"])
def test_rolling_groupby_multi(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [0, 0, 1, 1, 0, 1, 2, 1, 1, 0],
            "c": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        }
    )
    gdf = cudf.from_pandas(pdf)

    for window_size in range(1, len(pdf) + 1):
        expect = getattr(
            pdf.groupby(["a", "b"]).rolling(window_size), agg
        )().fillna(-1)
        got = getattr(
            gdf.groupby(["a", "b"]).rolling(window_size), agg
        )().fillna(-1)
        assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("agg", ["sum", "min", "max", "mean", "count"])
@pytest.mark.parametrize(
    "window_size", ["1d", "2d", "3d", "4d", "5d", "6d", "7d"]
)
def test_rolling_groupby_offset(agg, window_size):
    pdf = pd.DataFrame(
        {
            "date": pd.date_range(start="2016-01-01", periods=7, freq="D"),
            "group": [1, 2, 2, 1, 1, 2, 1],
            "val": [5, 6, 7, 8, 1, 2, 3],
        }
    ).set_index("date")
    gdf = cudf.from_pandas(pdf)
    expect = getattr(pdf.groupby("group").rolling(window_size), agg)().fillna(
        -1
    )
    got = getattr(gdf.groupby("group").rolling(window_size), agg)().fillna(-1)
    assert_eq(expect, got, check_dtype=False)
