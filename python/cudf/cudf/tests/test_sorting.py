# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import string
from itertools import product

import numpy as np
import pandas as pd
import pytest

from cudf import DataFrame, Series
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.core.column import NumericalColumn
from cudf.testing import assert_eq
from cudf.testing._utils import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
)

sort_nelem_args = [2, 257]
sort_dtype_args = [
    np.int32,
    np.int64,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
]
sort_slice_args = [slice(1, None), slice(None, -1), slice(1, -1)]


@pytest.mark.parametrize(
    "nelem,dtype", list(product(sort_nelem_args, sort_dtype_args))
)
def test_dataframe_sort_values(nelem, dtype):
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    df["a"] = aa = (100 * rng.random(nelem)).astype(dtype)
    df["b"] = bb = (100 * rng.random(nelem)).astype(dtype)
    sorted_df = df.sort_values(by="a")
    # Check
    sorted_index = np.argsort(aa, kind="mergesort")
    assert_eq(sorted_df.index.values, sorted_index)
    assert_eq(sorted_df["a"].values, aa[sorted_index])
    assert_eq(sorted_df["b"].values, bb[sorted_index])


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("index", ["a", "b", ["a", "b"]])
def test_dataframe_sort_values_ignore_index(index, ignore_index):
    if (
        PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION
        and isinstance(index, list)
        and not ignore_index
    ):
        pytest.skip(
            reason="Unstable sorting by pandas(numpy): https://github.com/pandas-dev/pandas/issues/57531"
        )

    gdf = DataFrame(
        {"a": [1, 3, 5, 2, 4], "b": [1, 1, 2, 2, 3], "c": [9, 7, 7, 7, 1]}
    )
    gdf = gdf.set_index(index)

    pdf = gdf.to_pandas()

    expect = pdf.sort_values(list(pdf.columns), ignore_index=ignore_index)
    got = gdf.sort_values((gdf.columns), ignore_index=ignore_index)

    assert_eq(expect, got)


@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_sort_values_ignore_index(ignore_index):
    gsr = Series([1, 3, 5, 2, 4])
    psr = gsr.to_pandas()

    expect = psr.sort_values(ignore_index=ignore_index)
    got = gsr.sort_values(ignore_index=ignore_index)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "nelem,sliceobj", list(product([10, 100], sort_slice_args))
)
def test_dataframe_sort_values_sliced(nelem, sliceobj):
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame()
    df["a"] = rng.random(nelem)

    expect = df[sliceobj]["a"].sort_values()
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj]["a"].sort_values()
    assert (got.to_pandas() == expect).all()


@pytest.mark.parametrize(
    "nelem,dtype,asc",
    list(product(sort_nelem_args, sort_dtype_args, [True, False])),
)
def test_series_argsort(nelem, dtype, asc):
    rng = np.random.default_rng(seed=0)
    sr = Series((100 * rng.random(nelem)).astype(dtype))
    res = sr.argsort(ascending=asc)

    if asc:
        expected = np.argsort(sr.to_numpy(), kind="mergesort")
    else:
        # -1 multiply works around missing desc sort (may promote to float64)
        expected = np.argsort(sr.to_numpy() * np.int8(-1), kind="mergesort")
    np.testing.assert_array_equal(expected, res.to_numpy())


@pytest.mark.parametrize(
    "nelem,asc", list(product(sort_nelem_args, [True, False]))
)
def test_series_sort_index(nelem, asc):
    rng = np.random.default_rng(seed=0)
    sr = Series(100 * rng.random(nelem))
    psr = sr.to_pandas()

    expected = psr.sort_index(ascending=asc)
    got = sr.sort_index(ascending=asc)

    assert_eq(expected, got)


@pytest.mark.parametrize("data", [[0, 1, 1, 2, 2, 2, 3, 3], [0], [1, 2, 3]])
@pytest.mark.parametrize("n", [-100, -50, -12, -2, 0, 1, 2, 3, 4, 7])
def test_series_nlargest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = Series(data)
    psr = pd.Series(data)
    assert_eq(sr.nlargest(n), psr.nlargest(n))
    assert_eq(sr.nlargest(n, keep="last"), psr.nlargest(n, keep="last"))

    assert_exceptions_equal(
        lfunc=psr.nlargest,
        rfunc=sr.nlargest,
        lfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
        rfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
    )


@pytest.mark.parametrize("data", [[0, 1, 1, 2, 2, 2, 3, 3], [0], [1, 2, 3]])
@pytest.mark.parametrize("n", [-100, -50, -12, -2, 0, 1, 2, 3, 4, 9])
def test_series_nsmallest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = Series(data)
    psr = pd.Series(data)
    assert_eq(sr.nsmallest(n), psr.nsmallest(n))
    assert_eq(
        sr.nsmallest(n, keep="last").sort_index(),
        psr.nsmallest(n, keep="last").sort_index(),
    )

    assert_exceptions_equal(
        lfunc=psr.nsmallest,
        rfunc=sr.nsmallest,
        lfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
        rfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
    )


@pytest.mark.parametrize("nelem,n", [(1, 1), (100, 100), (10, 5), (100, 10)])
@pytest.mark.parametrize("op", ["nsmallest", "nlargest"])
@pytest.mark.parametrize("columns", ["a", ["b", "a"]])
def test_dataframe_nlargest_nsmallest(nelem, n, op, columns):
    rng = np.random.default_rng(seed=0)
    aa = rng.random(nelem)
    bb = rng.random(nelem)

    df = DataFrame({"a": aa, "b": bb})
    pdf = df.to_pandas()
    assert_eq(getattr(df, op)(n, columns), getattr(pdf, op)(n, columns))


@pytest.mark.parametrize(
    "counts,sliceobj", list(product([(10, 5), (100, 10)], sort_slice_args))
)
def test_dataframe_nlargest_sliced(counts, sliceobj):
    nelem, n = counts
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame()
    df["a"] = rng.random(nelem)
    df["b"] = rng.random(nelem)

    expect = df[sliceobj].nlargest(n, "a")
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj].nlargest(n, "a")
    assert (got.to_pandas() == expect).all().all()


@pytest.mark.parametrize(
    "counts,sliceobj", list(product([(10, 5), (100, 10)], sort_slice_args))
)
def test_dataframe_nsmallest_sliced(counts, sliceobj):
    nelem, n = counts
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame()
    df["a"] = rng.random(nelem)
    df["b"] = rng.random(nelem)

    expect = df[sliceobj].nsmallest(n, "a")
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj].nsmallest(n, "a")
    assert (got.to_pandas() == expect).all().all()


@pytest.mark.parametrize("num_cols", [1, 2, 3, 5])
@pytest.mark.parametrize("num_rows", [0, 1, 2, 1000])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + DATETIME_TYPES)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column(
    num_cols, num_rows, dtype, ascending, na_position
):
    rng = np.random.default_rng(seed=0)
    by = list(string.ascii_lowercase[:num_cols])
    pdf = pd.DataFrame()

    for i in range(5):
        colname = string.ascii_lowercase[i]
        data = rng.integers(0, 26, num_rows).astype(dtype)
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got = gdf.sort_values(by, ascending=ascending, na_position=na_position)
    expect = pdf.sort_values(by, ascending=ascending, na_position=na_position)

    assert_eq(
        got[by].reset_index(drop=True), expect[by].reset_index(drop=True)
    )


@pytest.mark.parametrize("num_cols", [1, 2, 3])
@pytest.mark.parametrize("num_rows", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("nulls", ["some", "all"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column_nulls(
    num_cols, num_rows, dtype, nulls, ascending, na_position
):
    rng = np.random.default_rng(seed=0)
    by = list(string.ascii_lowercase[:num_cols])
    pdf = pd.DataFrame()

    for i in range(3):
        colname = string.ascii_lowercase[i]
        data = rng.integers(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = np.array([], dtype="int64")
            if num_rows > 0:
                idx = rng.choice(
                    num_rows, size=int(num_rows / 4), replace=False
                )
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got = gdf.sort_values(by, ascending=ascending, na_position=na_position)
    expect = pdf.sort_values(by, ascending=ascending, na_position=na_position)

    assert_eq(
        got[by].reset_index(drop=True), expect[by].reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "ascending", list(product((True, False), (True, False)))
)
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column_nulls_multiple_ascending(
    ascending, na_position
):
    pdf = pd.DataFrame(
        {"a": [3, 1, None, 2, 2, None, 1], "b": [1, 2, 3, 4, 5, 6, 7]}
    )
    gdf = DataFrame.from_pandas(pdf)
    expect = pdf.sort_values(
        by=["a", "b"], ascending=ascending, na_position=na_position
    )
    actual = gdf.sort_values(
        by=["a", "b"], ascending=ascending, na_position=na_position
    )

    assert_eq(actual, expect)


@pytest.mark.parametrize("nelem", [1, 100])
def test_series_nlargest_nelem(nelem):
    rng = np.random.default_rng(seed=0)
    elems = rng.random(nelem)
    gds = Series(elems).nlargest(nelem)
    pds = pd.Series(elems).nlargest(nelem)

    assert (pds == gds.to_pandas()).all().all()


@pytest.mark.parametrize("map_size", [1, 2, 8])
@pytest.mark.parametrize("nelem", [1, 10, 100])
@pytest.mark.parametrize("keep", [True, False])
def test_dataframe_scatter_by_map(map_size, nelem, keep):
    strlist = ["dog", "cat", "fish", "bird", "pig", "fox", "cow", "goat"]
    rng = np.random.default_rng(seed=0)
    df = DataFrame(
        {
            "a": rng.choice(strlist[:map_size], nelem),
            "b": rng.uniform(low=0, high=map_size, size=nelem),
            "c": rng.integers(map_size, size=nelem),
        }
    )
    df["d"] = df["a"].astype("category")

    def _check_scatter_by_map(dfs, col):
        assert len(dfs) == map_size
        nrows = 0
        # print(col._column)
        name = col.name
        for i, df in enumerate(dfs):
            nrows += len(df)
            if len(df) > 0:
                # Make sure the column types were preserved
                assert isinstance(df[name]._column, type(col._column))
            try:
                sr = df[name].astype(np.int32)
            except ValueError:
                sr = df[name]
            assert sr.nunique() <= 1
            if sr.nunique() == 1:
                if isinstance(df[name]._column, NumericalColumn):
                    assert sr.iloc[0] == i
        assert nrows == nelem

    with pytest.warns(UserWarning):
        _check_scatter_by_map(
            df.scatter_by_map("a", map_size, keep_index=keep), df["a"]
        )
    _check_scatter_by_map(
        df.scatter_by_map("b", map_size, keep_index=keep), df["b"]
    )
    _check_scatter_by_map(
        df.scatter_by_map("c", map_size, keep_index=keep), df["c"]
    )
    with pytest.warns(UserWarning):
        _check_scatter_by_map(
            df.scatter_by_map("d", map_size, keep_index=keep), df["d"]
        )

    if map_size == 2 and nelem == 100:
        with pytest.warns(UserWarning):
            df.scatter_by_map("a")  # Auto-detect map_size
        with pytest.raises(ValueError):
            with pytest.warns(UserWarning):
                df.scatter_by_map("a", map_size=1, debug=True)  # Bad map_size

    # Test Index
    df2 = df.set_index("c")
    generic_result = df2.scatter_by_map("b", map_size, keep_index=keep)
    _check_scatter_by_map(generic_result, df2["b"])
    if keep:
        for frame in generic_result:
            isinstance(frame.index, type(df2.index))

    # Test MultiIndex
    df2 = df.set_index(["a", "c"])
    multiindex_result = df2.scatter_by_map("b", map_size, keep_index=keep)
    _check_scatter_by_map(multiindex_result, df2["b"])
    if keep:
        for frame in multiindex_result:
            isinstance(frame.index, type(df2.index))


@pytest.mark.parametrize(
    "nelem,dtype", list(product(sort_nelem_args, sort_dtype_args))
)
@pytest.mark.parametrize(
    "kind", ["quicksort", "mergesort", "heapsort", "stable"]
)
def test_dataframe_sort_values_kind(nelem, dtype, kind):
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    df["a"] = aa = (100 * rng.random(nelem)).astype(dtype)
    df["b"] = bb = (100 * rng.random(nelem)).astype(dtype)
    with expect_warning_if(kind != "quicksort", UserWarning):
        sorted_df = df.sort_values(by="a", kind=kind)
    # Check
    sorted_index = np.argsort(aa, kind="mergesort")
    assert_eq(sorted_df.index.values, sorted_index)
    assert_eq(sorted_df["a"].values, aa[sorted_index])
    assert_eq(sorted_df["b"].values, bb[sorted_index])


@pytest.mark.parametrize("ids", [[-1, 0, 1, 0], [0, 2, 3, 0]])
def test_dataframe_scatter_by_map_7513(ids):
    df = DataFrame({"id": ids, "val": [0, 1, 2, 3]})
    with pytest.raises(ValueError):
        df.scatter_by_map(df["id"])


def test_dataframe_scatter_by_map_empty():
    df = DataFrame({"a": [], "b": []}, dtype="float64")
    scattered = df.scatter_by_map(df["a"])
    assert len(scattered) == 0


def test_sort_values_by_index_level():
    df = pd.DataFrame({"a": [1, 3, 2]}, index=pd.Index([1, 3, 2], name="b"))
    cudf_df = DataFrame.from_pandas(df)
    result = cudf_df.sort_values("b")
    expected = df.sort_values("b")
    assert_eq(result, expected)


def test_sort_values_by_ambiguous():
    df = pd.DataFrame({"a": [1, 3, 2]}, index=pd.Index([1, 3, 2], name="a"))
    cudf_df = DataFrame.from_pandas(df)

    assert_exceptions_equal(
        lfunc=df.sort_values,
        rfunc=cudf_df.sort_values,
        lfunc_args_and_kwargs=(["a"], {}),
        rfunc_args_and_kwargs=(["a"], {}),
    )
