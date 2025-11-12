# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import math
import pickle
import platform

import numpy as np
import pandas as pd
import pytest
from pandas.api.indexers import BaseIndexer

import cudf
from cudf.testing import assert_eq
from cudf.testing.dataset_generator import rand_dataframe


@pytest.fixture(params=[True, False])
def center(request):
    return request.param


@pytest.fixture
def supported_rolling_reductions(reduction_methods):
    if reduction_methods in [
        "product",
        "quantile",
        "all",
        "any",
        "median",
        "kurtosis",
        "skew",
    ]:
        pytest.skip(f"{reduction_methods} not implemented")
    return reduction_methods


@pytest.mark.parametrize(
    "data,index",
    [
        ([], []),
        ([1, 1, 1, 1], None),
        ([1, 2, 3, 4], pd.date_range("2001-01-01", "2001-01-04")),
        ([1, 2, 4, 9, 9, 4], ["a", "b", "c", "d", "e", "f"]),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "one", "some", "all"])
def test_rolling_series_basic(
    data, index, supported_rolling_reductions, nulls, center, request
):
    rng = np.random.default_rng(1)

    if len(data) > 0:
        if nulls == "one":
            p = rng.integers(0, len(data))
            data[p] = np.nan
        elif nulls == "some":
            p1, p2 = rng.integers(0, len(data), (2,))
            data[p1] = np.nan
            data[p2] = np.nan
        elif nulls == "all":
            data = [np.nan] * len(data)

    psr = pd.Series(data, index=index)
    gsr = cudf.from_pandas(psr)
    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            expect = getattr(
                psr.rolling(window_size, min_periods, center),
                supported_rolling_reductions,
            )().fillna(-1)
            got = getattr(
                gsr.rolling(window_size, min_periods, center),
                supported_rolling_reductions,
            )().fillna(-1)
            assert_eq(expect, got, check_dtype=False, check_freq=False)


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
@pytest.mark.parametrize("nulls", ["none", "one", "some", "all"])
def test_rolling_dataframe_basic(
    data, supported_rolling_reductions, nulls, center, request
):
    rng = np.random.default_rng(0)
    pdf = pd.DataFrame(data)

    if len(pdf) > 0:
        if nulls == "all":
            pdf = pd.DataFrame(np.nan, columns=pdf.columns, index=pdf.index)
        else:
            for col_idx in range(len(pdf.columns)):
                if nulls == "one":
                    p = rng.integers(0, len(data))
                    pdf.iloc[p, col_idx] = np.nan
                elif nulls == "some":
                    p1, p2 = rng.integers(0, len(data), (2,))
                    pdf.iloc[p1, col_idx] = np.nan
                    pdf.iloc[p2, col_idx] = np.nan

    gdf = cudf.from_pandas(pdf)
    for window_size in range(1, len(data) + 1):
        for min_periods in range(1, window_size + 1):
            expect = getattr(
                pdf.rolling(window_size, min_periods, center),
                supported_rolling_reductions,
            )().fillna(-1)
            got = getattr(
                gdf.rolling(window_size, min_periods, center),
                supported_rolling_reductions,
            )().fillna(-1)
            assert_eq(expect, got, check_dtype=False)


def test_rolling_with_offset(supported_rolling_reductions):
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
        getattr(psr.rolling("2s"), supported_rolling_reductions)().fillna(-1),
        getattr(gsr.rolling("2s"), supported_rolling_reductions)().fillna(-1),
        check_dtype=False,
    )


@pytest.mark.parametrize("agg", ["std", "var"])
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("window_size", [2, 100])
def test_rolling_var_std_large(agg, ddof, center, window_size):
    iupper_bound = math.sqrt(np.iinfo(np.int64).max / window_size)
    ilower_bound = -math.sqrt(abs(np.iinfo(np.int64).min) / window_size)

    fupper_bound = math.sqrt(np.finfo(np.float64).max / window_size)
    flower_bound = -math.sqrt(abs(np.finfo(np.float64).min) / window_size)

    n_rows = 1_000
    data = rand_dataframe(
        dtypes_meta=[
            {
                "dtype": "int64",
                "null_frequency": 0.4,
                "cardinality": n_rows,
                "min_bound": ilower_bound,
                "max_bound": iupper_bound,
            },
            {
                "dtype": "float64",
                "null_frequency": 0.4,
                "cardinality": n_rows,
                "min_bound": flower_bound,
                "max_bound": fupper_bound,
            },
            {
                "dtype": "decimal64",
                "null_frequency": 0.4,
                "cardinality": n_rows,
                "min_bound": ilower_bound,
                "max_bound": iupper_bound,
            },
        ],
        rows=n_rows,
        use_threads=False,
        seed=100,
    )
    pdf = data.to_pandas()
    gdf = cudf.from_pandas(pdf)

    expect = getattr(pdf.rolling(window_size, 1, center), agg)(ddof=ddof)
    got = getattr(gdf.rolling(window_size, 1, center), agg)(ddof=ddof)

    if platform.machine() == "aarch64":
        # Due to pandas-37051, pandas rolling var/std on uniform window is
        # not reliable. Skipping these rows when comparing.
        for col in expect:
            mask = (got[col].fillna(-1) != 0).to_pandas()
            expect[col] = expect[col][mask]
            got[col] = got[col][mask]
            assert_eq(expect[col], got[col], check_freq=False)
    else:
        assert_eq(expect, got, check_freq=False)


def test_rolling_var_uniform_window():
    """
    Pandas adopts an online variance calculation algorithm. This gives a
    floating point artifact.

    In cudf, each window is computed independently from the previous window,
    this gives better numeric precision.
    """

    s = pd.Series([1e8, 5, 5, 5])
    expected = s.rolling(3).var()
    got = cudf.from_pandas(s).rolling(3).var()

    assert_eq(expected, got)


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
    index = pd.DatetimeIndex(
        pd.date_range("2000-01-01", "2000-01-02", freq="1h")
    )
    pdf = pd.DataFrame({"x": np.arange(len(index))}, index=index)
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.rolling("2h").x.mean(),
        gdf.rolling("2h").x.mean(),
        check_freq=False,
    )


@pytest.mark.parametrize(
    "data,index", [([1.2, 4.5, 5.9, 2.4, 9.3, 7.1], None), ([], [])]
)
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
def test_rolling_dataframe_numba_udf_basic(data, center):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    def some_func(A):
        b = 0
        for a in A:
            b = b + a**2
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


def test_rolling_groupby_simple(supported_rolling_reductions):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        }
    )
    gdf = cudf.from_pandas(pdf)

    for window_size in range(1, len(pdf) + 1):
        expect = getattr(
            pdf.groupby("a").rolling(window_size), supported_rolling_reductions
        )().fillna(-1)
        got = getattr(
            gdf.groupby("a").rolling(window_size), supported_rolling_reductions
        )().fillna(-1)
        assert_eq(expect, got, check_dtype=False)

    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 1, 2, 2, 3], "c": [1, 2, 3, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    for window_size in range(1, len(pdf) + 1):
        expect = getattr(
            pdf.groupby("a").rolling(window_size), supported_rolling_reductions
        )().fillna(-1)
        got = getattr(
            gdf.groupby("a").rolling(window_size), supported_rolling_reductions
        )().fillna(-1)
        assert_eq(expect, got, check_dtype=False)


def test_rolling_groupby_multi(supported_rolling_reductions):
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
            pdf.groupby(["a", "b"], sort=True).rolling(window_size),
            supported_rolling_reductions,
        )().fillna(-1)
        got = getattr(
            gdf.groupby(["a", "b"], sort=True).rolling(window_size),
            supported_rolling_reductions,
        )().fillna(-1)
        assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("window_size", ["1d", "3d", "6d", "7d"])
def test_rolling_groupby_offset(supported_rolling_reductions, window_size):
    pdf = pd.DataFrame(
        {
            "date": pd.date_range(start="2016-01-01", periods=7, freq="D"),
            "group": [1, 2, 2, 1, 1, 2, 1],
            "val": [5, 6, 7, 8, 1, 2, 3],
        }
    ).set_index("date")
    gdf = cudf.from_pandas(pdf)
    expect = getattr(
        pdf.groupby("group").rolling(window_size), supported_rolling_reductions
    )().fillna(-1)
    got = getattr(
        gdf.groupby("group").rolling(window_size), supported_rolling_reductions
    )().fillna(-1)
    assert_eq(expect, got, check_dtype=False)


def test_rolling_custom_index_support():
    class CustomIndexer(BaseIndexer):
        def get_window_bounds(
            self, num_values, min_periods, center, closed, step=None
        ):
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
        pd.api.indexers.VariableOffsetWindowIndexer(
            index=pd.date_range("2020", periods=5), offset=pd.offsets.BDay(1)
        ),
    ],
)
def test_rolling_indexer_support(indexer):
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    gdf = cudf.from_pandas(df)

    expected = df.rolling(window=indexer, min_periods=2).sum()
    actual = gdf.rolling(window=indexer, min_periods=2).sum()

    assert_eq(expected, actual)


def test_rolling_series():
    df = cudf.DataFrame({"a": range(0, 100), "b": [10, 20, 30, 40, 50] * 20})
    pdf = df.to_pandas()

    expected = pdf.groupby("b")["a"].rolling(5).mean()
    actual = df.groupby("b")["a"].rolling(5).mean()

    assert_eq(expected, actual)


@pytest.mark.parametrize("klass", ["DataFrame", "Series"])
def test_pandas_compat_int_nan_min_periods(klass):
    data = [None, 1, 2, None, 4, 6, 11]
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(cudf, klass)(data).rolling(2, min_periods=1).sum()
    expected = getattr(pd, klass)(data).rolling(2, min_periods=1).sum()
    assert_eq(result, expected)

    result = getattr(cudf, klass)(data).rolling(2, min_periods=1).sum()
    expected = getattr(cudf, klass)([None, 1, 3, 2, 4, 10, 17])
    assert_eq(result, expected)


def test_groupby_rolling_pickleable():
    df = cudf.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    gb_rolling = pickle.loads(pickle.dumps(df.groupby("a").rolling(2)))
    assert_eq(gb_rolling.obj, cudf.DataFrame({"b": [1, 2, 3]}))
