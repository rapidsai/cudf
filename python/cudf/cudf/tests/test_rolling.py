# Copyright (c) 2021, NVIDIA CORPORATION.

import math

import numpy as np
import pandas as pd
import pytest

import cudf
import cudf.testing.dataset_generator as dataset_generator
from cudf.core._compat import PANDAS_GE_110
from cudf.testing._utils import assert_eq


@pytest.mark.parametrize(
    "data,index",
    [
        ([], []),
        ([1, 1, 1, 1], None),
        ([1, 2, 3, 4], pd.date_range("2001-01-01", "2001-01-04")),
        ([1, 2, 4, 9, 9, 4], ["a", "b", "c", "d", "e", "f"]),
    ],
)
@pytest.mark.parametrize(
    "agg", ["sum", "min", "max", "mean", "count", "std", "var"]
)
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

    psr = cudf.utils.utils._create_pandas_series(data=data, index=index)
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
@pytest.mark.parametrize(
    "agg", ["sum", "min", "max", "mean", "count", "std", "var"]
)
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
        pytest.param("count"),
        pytest.param("std"),
        pytest.param("var"),
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


@pytest.mark.parametrize("agg", ["std", "var"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("seed", [100, 1000, 10000])
@pytest.mark.parametrize("window_size", [2, 10, 100, 1000])
def test_rolling_var_std_large(agg, center, seed, window_size):
    if PANDAS_GE_110:
        kwargs = {"check_freq": False}
    else:
        kwargs = {}

    n_rows = 1_000
    data = dataset_generator.rand_dataframe(
        dtypes_meta=[
            {"dtype": "i4", "null_frequency": 0.4, "cardinality": 100},
            {"dtype": "f8", "null_frequency": 0.4, "cardinality": 100},
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 100},
            {"dtype": "decimal32", "null_frequency": 0.4, "cardinality": 100},
        ],
        rows=n_rows,
        use_threads=False,
        seed=seed,
    )

    pdf = data.to_pandas()
    gdf = cudf.from_pandas(pdf)

    expect = getattr(pdf.rolling(window_size, 1, center), agg)().fillna(-1)
    got = getattr(gdf.rolling(window_size, 1, center), agg)().fillna(-1)

    # Pandas adopts an online variance calculation algorithm. Each window has a
    # numeric error from the previous window. This makes the variance of a
    # uniform window has a small residue. Taking the square root of a very
    # small number may result in a non-trival number.
    #
    # In cudf, each window is computed independently from the previous window,
    # this gives better numeric precision.
    #
    # For quantitative analysis:
    # https://gist.github.com/isVoid/4984552da6ef5545348399c22d72cffb
    #
    # To make up this difference, we skip comparing uniform windows by coercing
    # pandas result of these windows to 0.
    for col in ["1", "2", "3"]:
        expect[col][(got[col] == 0.0).to_pandas()] = 0.0

    assert_eq(expect, got, **kwargs)


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

    psr = cudf.utils.utils._create_pandas_series(data=data, index=index)
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


@pytest.mark.parametrize(
    "agg", ["sum", "min", "max", "mean", "count", "var", "std"]
)
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

    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 1, 2, 2, 3], "c": [1, 2, 3, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    for window_size in range(1, len(pdf) + 1):
        expect = getattr(pdf.groupby("a").rolling(window_size), agg)().fillna(
            -1
        )
        got = getattr(gdf.groupby("a").rolling(window_size), agg)().fillna(-1)
        assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "agg", ["sum", "min", "max", "mean", "count", "var", "std"]
)
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
            pdf.groupby(["a", "b"], sort=True).rolling(window_size), agg
        )().fillna(-1)
        got = getattr(
            gdf.groupby(["a", "b"], sort=True).rolling(window_size), agg
        )().fillna(-1)
        assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "agg", ["sum", "min", "max", "mean", "count", "var", "std"]
)
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


def test_rolling_custom_index_support():
    from pandas.api.indexers import BaseIndexer

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)

            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size

            return start, end

    use_expanding = [True, False, True, False, True]
    indexer = CustomIndexer(window_size=1, use_expanding=use_expanding)

    df = pd.DataFrame({"values": range(5)})
    gdf = cudf.from_pandas(df)

    expected = df.rolling(window=indexer).sum()
    actual = gdf.rolling(window=indexer).sum()

    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "indexer",
    [
        pd.api.indexers.FixedForwardWindowIndexer(window_size=2),
        pd.core.window.indexers.ExpandingIndexer(),
        pd.core.window.indexers.FixedWindowIndexer(window_size=3),
    ],
)
def test_rolling_indexer_support(indexer):
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    gdf = cudf.from_pandas(df)

    expected = df.rolling(window=indexer, min_periods=2).sum()
    actual = gdf.rolling(window=indexer, min_periods=2).sum()

    assert_eq(expected, actual)
