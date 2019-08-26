import math

import numpy as np
import pandas as pd
import pytest

import cudf
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
def test_rollling_series_basic(data, index, agg, nulls, center):
    if len(data) > 0:
        if nulls == "one":
            p = np.random.randint(0, len(data))
            data[p] = None
        elif nulls == "some":
            p1, p2 = np.random.randint(0, len(data), (2,))
            data[p1] = None
            data[p2] = None
        elif nulls == "all":
            data = [None] * len(data)

    psr = pd.Series(data, index=index)
    gsr = cudf.from_pandas(psr)

    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            assert_eq(
                getattr(
                    psr.rolling(window_size, min_periods, center), agg
                )().fillna(-1),
                getattr(
                    gsr.rolling(window_size, min_periods, center), agg
                )().fillna(-1),
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
@pytest.mark.parametrize("agg", ["sum", "min", "max", "mean", "count"])
@pytest.mark.parametrize("nulls", ["none", "one", "some", "all"])
@pytest.mark.parametrize("center", [True, False])
def test_rolling_dataframe_basic(data, agg, nulls, center):
    pdf = pd.DataFrame(data)

    if len(pdf) > 0:
        for col_name in pdf.columns:
            if nulls == "one":
                p = np.random.randint(0, len(data))
                pdf[col_name][p] = None
            elif nulls == "some":
                p1, p2 = np.random.randint(0, len(data), (2,))
                pdf[col_name][p1] = None
                pdf[col_name][p2] = None
            elif nulls == "all":
                pdf[col_name][:] = None

    gdf = cudf.from_pandas(pdf)

    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            assert_eq(
                getattr(
                    pdf.rolling(window_size, min_periods, center), agg
                )().fillna(-1),
                getattr(
                    gdf.rolling(window_size, min_periods, center), agg
                )().fillna(-1),
                check_dtype=False,
            )


@pytest.mark.parametrize(
    "agg",
    [
        "sum",
        pytest.param(
            "min", marks=pytest.mark.xfail(reason="Pandas bug fixed in 0.24.2")
        ),
        pytest.param(
            "max", marks=pytest.mark.xfail(reason="Pandas bug fixed in 0.24.2")
        ),
        "mean",
        "count",
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
    index = pd.DatetimeIndex(start="2000-01-01", end="2000-01-02", freq="1h")
    pdf = pd.DataFrame({"x": np.arange(len(index))}, index=index)
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.rolling("2h").x.mean(), gdf.rolling("2h").x.mean())


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
