# Copyright (c) 2025, NVIDIA CORPORATION.

import itertools
import operator
import pickle
from contextlib import contextmanager
from io import BytesIO

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.column import as_column
from cudf.testing import assert_eq, assert_neq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@contextmanager
def expect_pandas_performance_warning(idx):
    with expect_warning_if(
        (not isinstance(idx[0], tuple) and len(idx) > 2)
        or (isinstance(idx[0], tuple) and len(idx[0]) > 2),
        pd.errors.PerformanceWarning,
    ):
        yield


def test_multiindex_df_assignment():
    pdf = pd.DataFrame({"x": [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)
    pdf.index = pd.MultiIndex([["a", "b"], ["c", "d"]], [[0, 1, 0], [1, 0, 1]])
    gdf.index = cudf.MultiIndex(
        levels=[["a", "b"], ["c", "d"]], codes=[[0, 1, 0], [1, 0, 1]]
    )
    assert_eq(pdf, gdf)


def test_multiindex_series_assignment():
    ps = pd.Series([1, 2, 3])
    gs = cudf.from_pandas(ps)
    ps.index = pd.MultiIndex([["a", "b"], ["c", "d"]], [[0, 1, 0], [1, 0, 1]])
    gs.index = cudf.MultiIndex(
        levels=[["a", "b"], ["c", "d"]], codes=[[0, 1, 0], [1, 0, 1]]
    )
    assert_eq(ps, gs)


def test_string_index():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 5)))
    gdf = cudf.from_pandas(pdf)
    stringIndex = ["a", "b", "c", "d", "e"]
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = np.array(["a", "b", "c", "d", "e"])
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = cudf.Index(["a", "b", "c", "d", "e"], name="name")
    pdf.index = stringIndex.to_pandas()
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = cudf.Index._from_column(
        as_column(["a", "b", "c", "d", "e"]), name="name"
    )
    pdf.index = stringIndex.to_pandas()
    gdf.index = stringIndex
    assert_eq(pdf, gdf)


def test_multiindex_row_shape():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(0, 5)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([["a", "b", "c"]], [[0]])
    pdfIndex.names = ["alpha"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)

    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([], {"a": pdf, "b": "index", "c": pdfIndex}),
        rfunc_args_and_kwargs=([], {"a": gdf, "b": "index", "c": gdfIndex}),
    )


@pytest.fixture
def pdf():
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame(rng.random(size=(7, 5)))


@pytest.fixture
def gdf(pdf):
    return cudf.from_pandas(pdf)


@pytest.fixture
def pdfIndex():
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    return pdfIndex


@pytest.fixture
def pdfIndexNulls():
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
        ],
        [
            [0, 0, 0, -1, 1, 1, 2],
            [1, -1, 1, 1, 0, 0, -1],
            [-1, 0, 2, 2, 2, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather"]
    return pdfIndex


def test_from_pandas(pdf, pdfIndex):
    pdf = pdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf, gdf)


def test_from_pandas_series():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    ).set_index(["a", "b"])

    result = cudf.from_pandas(pdf)
    assert_eq(pdf, result)

    test_pdf = pdf["c"]
    result = cudf.from_pandas(test_pdf)
    assert_eq(test_pdf, result)


def test_series_multiindex(pdfIndex):
    rng = np.random.default_rng(seed=0)
    ps = pd.Series(rng.random(7))
    gs = cudf.from_pandas(ps)
    ps.index = pdfIndex
    gs.index = cudf.from_pandas(pdfIndex)
    assert_eq(ps, gs)


def test_multiindex_getitem(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.index[0], gdf.index[0])


@pytest.mark.parametrize(
    "key_tuple",
    [
        # return 2 rows, 0 remaining keys = dataframe with entire index
        ("a", "store", "clouds", "fire"),
        (("a", "store", "clouds", "fire"), slice(None)),
        # return 2 rows, 1 remaining key = dataframe with n-k index columns
        ("a", "store", "storm"),
        (("a", "store", "storm"), slice(None)),
        # return 2 rows, 2 remaining keys = dataframe with n-k index columns
        ("a", "store"),
        (("a", "store"), slice(None)),
        # return 2 rows, n-1 remaining keys = dataframe with n-k index columns
        ("a",),
        "a",
        "b",
        "c",
        (("a",), slice(None)),
        # return 1 row, 0 remaining keys = dataframe with entire index
        ("a", "store", "storm", "smoke"),
        (("a", "store", "storm", "smoke"), slice(None)),
        # return 1 row and 1 remaining key = series
        ("c", "forest", "clear"),
        (("c", "forest", "clear"), slice(None)),
    ],
)
def test_multiindex_loc(pdf, gdf, pdfIndex, key_tuple):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with expect_pandas_performance_warning(key_tuple):
        expected = pdf.loc[key_tuple]
    got = gdf.loc[key_tuple].sort_index()
    assert_eq(expected.sort_index(), got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gdf.loc[key_tuple]
    assert_eq(expected, got)


@pytest.mark.parametrize("second_val", [[0, 1], [1, 0]])
def test_multiindex_compatible_ordering(second_val):
    indexer = (([1, 1], second_val), slice(None))
    df = pd.DataFrame(
        {"a": [1, 1, 2, 3], "b": [1, 0, 1, 1], "c": [1, 2, 3, 4]}
    ).set_index(["a", "b"])
    cdf = cudf.from_pandas(df)
    expect = df.loc[indexer]
    with cudf.option_context("mode.pandas_compatible", True):
        actual = cdf.loc[indexer]
    assert_eq(actual, expect)


@pytest.mark.parametrize(
    "arg",
    [
        slice(("a", "store"), ("b", "house")),
        slice(None, ("b", "house")),
        slice(("a", "store"), None),
        slice(None),
    ],
)
def test_multiindex_loc_slice(pdf, pdfIndex, arg):
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[arg], gdf.loc[arg])


def test_multiindex_loc_errors(pdf, pdfIndex):
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf.index = gdfIndex

    with pytest.raises(KeyError):
        gdf.loc[("a", "store", "clouds", "foo")]
    with pytest.raises(IndexError):
        gdf.loc[
            ("a", "store", "clouds", "fire", "x", "y")
        ]  # too many indexers
    with pytest.raises(IndexError):
        gdf.loc[slice(None, ("a", "store", "clouds", "fire", "x", "y"))]


def test_multiindex_loc_then_column(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with pytest.warns(pd.errors.PerformanceWarning):
        expected = pdf.loc[("a", "store", "clouds", "fire"), :][0]
    got = gdf.loc[("a", "store", "clouds", "fire"), :][0]
    assert_eq(expected, got)


def test_multiindex_loc_rows_0(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex

    assert_exceptions_equal(
        lfunc=pdf.loc.__getitem__,
        rfunc=gdf.loc.__getitem__,
        lfunc_args_and_kwargs=([(("d",), slice(None, None, None))],),
        rfunc_args_and_kwargs=([(("d",), slice(None, None, None))],),
    )


def test_multiindex_loc_rows_1_2_key(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[("c", "forest"), :], gdf.loc[("c", "forest"), :])


def test_multiindex_loc_rows_1_1_key(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[("c",), :], gdf.loc[("c",), :])


def test_multiindex_column_shape():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 0)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([["a", "b", "c"]], [[0]])
    pdfIndex.names = ["alpha"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)

    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([], {"a": pdf, "b": "columns", "c": pdfIndex}),
        rfunc_args_and_kwargs=([], {"a": gdf, "b": "columns", "c": gdfIndex}),
    )


@pytest.mark.parametrize(
    "query",
    [
        ("a", "store", "clouds", "fire"),
        ("a", "store", "storm", "smoke"),
        ("a", "store"),
        ("b", "house"),
        ("a", "store", "storm"),
        ("a",),
        ("c", "forest", "clear"),
    ],
)
def test_multiindex_columns(pdf, pdfIndex, query):
    pdf = pdf.copy(deep=False)
    pdf = pdf.T
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.columns = pdfIndex
    gdf.columns = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with expect_pandas_performance_warning(query):
        expected = pdf[query]
    got = gdf[query]
    assert_eq(expected, got)


def test_multiindex_index_and_columns():
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "x": rng.integers(0, 5, 5),
            "y": rng.integers(0, 5, 5),
        }
    )
    pdf = gdf.to_pandas()
    mi = cudf.MultiIndex(
        levels=[[0, 1, 2], [3, 4]],
        codes=[[0, 0, 1, 1, 2], [0, 1, 0, 1, 1]],
        names=["x", "y"],
    )
    gdf.index = mi
    mc = cudf.MultiIndex(
        levels=[["val"], ["mean", "min"]], codes=[[0, 0], [0, 1]]
    )
    gdf.columns = mc
    pdf.index = mi.to_pandas()
    pdf.columns = mc.to_pandas()
    assert_eq(pdf, gdf)


def test_multiindex_multiple_groupby():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": [4, 17, 4, 9, 5],
            "b": [1, 4, 4, 3, 2],
            "x": rng.normal(size=5),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    pdg = pdf.groupby(["a", "b"], sort=True).sum()
    gdg = gdf.groupby(["a", "b"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["a", "b"], sort=True).x.sum()
    gdg = gdf.groupby(["a", "b"], sort=True).x.sum()
    assert_eq(pdg, gdg)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby(["x", "y"], sort=True).z.sum(),
        lambda df: df.groupby(["x", "y"], sort=True).sum(),
    ],
)
def test_multi_column(func):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=1000),
            "y": rng.integers(0, 10, size=1000),
            "z": rng.normal(size=1000),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    a = func(pdf)
    b = func(gdf)

    assert_eq(a, b)


def test_multiindex_equality():
    # mi made from groupby
    # mi made manually to be identical
    # are they equal?
    gdf = cudf.DataFrame(
        {"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5], "z": [0, 1, 0, 1, 0]}
    )
    mi1 = gdf.groupby(["x", "y"], sort=True).mean().index
    mi2 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    assert_eq(mi1, mi2)

    # mi made from two groupbys, are they equal?
    mi2 = gdf.groupby(["x", "y"], sort=True).max().index
    assert_eq(mi1, mi2)

    # mi made manually twice are they equal?
    mi1 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    mi2 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    assert_eq(mi1, mi2)

    # mi made from different groupbys are they not equal?
    mi1 = gdf.groupby(["x", "y"]).mean().index
    mi2 = gdf.groupby(["x", "z"]).mean().index
    assert_neq(mi1, mi2)

    # mi made from different manuals are they not equal?
    mi1 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    mi2 = cudf.MultiIndex(
        levels=[[0, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    assert_neq(mi1, mi2)


def test_multiindex_equals():
    # mi made from groupby
    # mi made manually to be identical
    # are they equal?
    gdf = cudf.DataFrame(
        {"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5], "z": [0, 1, 0, 1, 0]}
    )
    mi1 = gdf.groupby(["x", "y"], sort=True).mean().index
    mi2 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    assert_eq(mi1.equals(mi2), True)

    # mi made from two groupbys, are they equal?
    mi2 = gdf.groupby(["x", "y"], sort=True).max().index
    assert_eq(mi1.equals(mi2), True)

    # mi made manually twice are they equal?
    mi1 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    mi2 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    assert_eq(mi1.equals(mi2), True)

    # mi made from different groupbys are they not equal?
    mi1 = gdf.groupby(["x", "y"], sort=True).mean().index
    mi2 = gdf.groupby(["x", "z"], sort=True).mean().index
    assert_eq(mi1.equals(mi2), False)

    # mi made from different manuals are they not equal?
    mi1 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    mi2 = cudf.MultiIndex(
        levels=[[0, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    assert_eq(mi1.equals(mi2), False)


@pytest.mark.parametrize(
    "iloc_rows",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
@pytest.mark.parametrize(
    "iloc_columns",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
def test_multiindex_iloc(pdf, gdf, pdfIndex, iloc_rows, iloc_columns):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    presult = pdf.iloc[iloc_rows, iloc_columns]
    gresult = gdf.iloc[iloc_rows, iloc_columns]
    if isinstance(gresult, cudf.DataFrame):
        assert_eq(
            presult, gresult, check_index_type=False, check_column_type=False
        )
    else:
        assert_eq(presult, gresult, check_index_type=False, check_dtype=False)


def test_multiindex_iloc_scalar():
    arrays = [["a", "a", "b", "b"], [1, 2, 3, 4]]
    tuples = list(zip(*arrays, strict=True))
    idx = cudf.MultiIndex.from_tuples(tuples)
    gdf = cudf.DataFrame(
        {"first": cp.random.rand(4), "second": cp.random.rand(4)}
    )
    gdf.index = idx

    pdf = gdf.to_pandas()
    assert_eq(pdf.iloc[3], gdf.iloc[3])


@pytest.mark.parametrize(
    "iloc_rows",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
@pytest.mark.parametrize(
    "iloc_columns",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
def test_multicolumn_iloc(pdf, gdf, pdfIndex, iloc_rows, iloc_columns):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    pdf = pdf.T
    gdf = gdf.T
    presult = pdf.iloc[iloc_rows, iloc_columns]
    gresult = gdf.iloc[iloc_rows, iloc_columns]
    if hasattr(gresult, "name") and isinstance(gresult.name, tuple):
        name = gresult.name[len(gresult.name) - 1]
        if isinstance(name, str) and "cudf" in name:
            gresult.name = name
    if isinstance(presult, pd.DataFrame):
        assert_eq(
            presult, gresult, check_index_type=False, check_column_type=False
        )
    else:
        assert_eq(presult, gresult, check_index_type=False, check_dtype=False)


def test_multicolumn_item():
    gdf = cudf.DataFrame(
        {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10)}
    )
    gdg = gdf.groupby(["x", "y"]).min()
    gdgT = gdg.T
    pdgT = gdgT.to_pandas()
    assert_eq(gdgT[(0, 0)], pdgT[(0, 0)])


def test_multiindex_reset_index(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.reset_index(), gdf.reset_index())


def test_multiindex_groupby_reset_index():
    gdf = cudf.DataFrame(
        {"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5], "z": [0, 1, 0, 1, 0]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["x", "y"], sort=True).sum()
    pdg = pdf.groupby(["x", "y"], sort=True).sum()
    assert_eq(pdg.reset_index(), gdg.reset_index())


def test_multicolumn_reset_index():
    gdf = cudf.DataFrame({"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5]})
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["x"], sort=True).agg({"y": ["count", "mean"]})
    pdg = pdf.groupby(["x"], sort=True).agg({"y": ["count", "mean"]})
    assert_eq(pdg.reset_index(), gdg.reset_index(), check_dtype=False)
    gdg = gdf.groupby(["x"], sort=True).agg({"y": ["count"]})
    pdg = pdf.groupby(["x"], sort=True).agg({"y": ["count"]})
    assert_eq(pdg.reset_index(), gdg.reset_index(), check_dtype=False)
    gdg = gdf.groupby(["x"], sort=True).agg({"y": "count"})
    pdg = pdf.groupby(["x"], sort=True).agg({"y": "count"})
    assert_eq(pdg.reset_index(), gdg.reset_index(), check_dtype=False)


def test_multiindex_multicolumn_reset_index():
    gdf = cudf.DataFrame(
        {"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5], "z": [1, 2, 3, 4, 5]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["x", "y"], sort=True).agg({"y": ["count", "mean"]})
    pdg = pdf.groupby(["x", "y"], sort=True).agg({"y": ["count", "mean"]})
    assert_eq(pdg.reset_index(), gdg.reset_index(), check_dtype=False)
    gdg = gdf.groupby(["x", "z"], sort=True).agg({"y": ["count", "mean"]})
    pdg = pdf.groupby(["x", "z"], sort=True).agg({"y": ["count", "mean"]})
    assert_eq(pdg.reset_index(), gdg.reset_index(), check_dtype=False)


def test_groupby_multiindex_columns_from_pandas(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(gdf, pdf)
    assert_eq(gdf.T, pdf.T)


def test_multiindex_rows_with_wildcard(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    gdf = gdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with pytest.warns(pd.errors.PerformanceWarning):
        assert_eq(
            pdf.loc[("a",), :].sort_index(), gdf.loc[("a",), :].sort_index()
        )
        assert_eq(
            pdf.loc[(("a"), ("store")), :].sort_index(),
            gdf.loc[(("a"), ("store")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(("a"), ("store"), ("storm")), :].sort_index(),
            gdf.loc[(("a"), ("store"), ("storm")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(("a"), ("store"), ("storm"), ("smoke")), :].sort_index(),
            gdf.loc[(("a"), ("store"), ("storm"), ("smoke")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(slice(None), "store"), :].sort_index(),
            gdf.loc[(slice(None), "store"), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(slice(None), slice(None), "storm"), :].sort_index(),
            gdf.loc[(slice(None), slice(None), "storm"), :].sort_index(),
        )
        assert_eq(
            pdf.loc[
                (slice(None), slice(None), slice(None), "smoke"), :
            ].sort_index(),
            gdf.loc[
                (slice(None), slice(None), slice(None), "smoke"), :
            ].sort_index(),
        )


def test_multiindex_multicolumn_zero_row_slice():
    gdf = cudf.DataFrame(
        {"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5], "z": [1, 2, 3, 4, 5]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["x", "y"]).agg({"z": ["count"]}).iloc[:0]
    pdg = pdf.groupby(["x", "y"]).agg({"z": ["count"]}).iloc[:0]
    assert_eq(pdg, gdg, check_dtype=False)


def test_multicolumn_loc(pdf, pdfIndex):
    pdf = pdf.copy(deep=False)
    pdf = pdf.T
    pdf.columns = pdfIndex
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.loc[:, "a"], gdf.loc[:, "a"])
    assert_eq(pdf.loc[:, ("a", "store")], gdf.loc[:, ("a", "store")])
    assert_eq(pdf.loc[:, "a":"b"], gdf.loc[:, "a":"b"])
    assert_eq(pdf.loc[:, ["a", "b"]], gdf.loc[:, ["a", "b"]])


@pytest.mark.xfail(
    reason="https://github.com/pandas-dev/pandas/issues/43351",
)
def test_multicolumn_set_item(pdf, pdfIndex):
    pdf = pdf.copy(deep=False)
    pdf = pdf.T
    pdf.columns = pdfIndex
    gdf = cudf.from_pandas(pdf)
    pdf["d"] = [1, 2, 3, 4, 5]
    gdf["d"] = [1, 2, 3, 4, 5]
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "key",
    [0, 1, [], [0, 1], slice(None), slice(0, 0), slice(0, 1), slice(0, 2)],
)
def test_multiindex_indexing(key):
    gi = cudf.MultiIndex.from_frame(
        cudf.DataFrame({"a": [1, 2, 3], "b": [True, False, False]})
    )
    pi = gi.to_pandas()

    assert_eq(gi[key], pi[key], exact=False)


@pytest.mark.parametrize(
    "names",
    [
        ["a", "b", "c"],
        [None, None, None],
        ["aa", "aa", "aa"],
        ["bb", "aa", "aa"],
        None,
    ],
)
def test_pickle_roundtrip_multiindex(names):
    df = cudf.DataFrame(
        {
            "one": [1, 2, 3],
            "two": [True, False, True],
            "three": ["ab", "cd", "ef"],
            "four": [0.2, 0.1, -10.2],
        }
    )
    expected_df = df.set_index(["one", "two", "three"])
    expected_df.index.names = names
    local_file = BytesIO()

    pickle.dump(expected_df, local_file)
    local_file.seek(0)
    actual_df = pickle.load(local_file)
    assert_eq(expected_df, actual_df)


def test_multiindex_index_single_row():
    arrays = [["a", "a", "b", "b"], [1, 2, 3, 4]]
    tuples = list(zip(*arrays, strict=True))
    idx = cudf.MultiIndex.from_tuples(tuples)
    gdf = cudf.DataFrame(
        {"first": cp.random.rand(4), "second": cp.random.rand(4)}
    )
    gdf.index = idx
    pdf = gdf.to_pandas()
    assert_eq(pdf.loc[("b", 3)], gdf.loc[("b", 3)])


@pytest.mark.parametrize(
    "levels",
    itertools.chain.from_iterable(
        itertools.permutations(range(3), n) for n in range(1, 4)
    ),
    ids=str,
)
def test_multiindex_sort_index_partial(levels):
    df = pd.DataFrame(
        {
            "a": [3, 3, 3, 1, 1, 1, 2, 2],
            "b": [4, 2, 7, -1, 11, -2, 7, 7],
            "c": [4, 4, 2, 3, 3, 3, 1, 1],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    ).set_index(["a", "b", "c"])
    cdf = cudf.from_pandas(df)

    expect = df.sort_index(level=levels, sort_remaining=True)
    got = cdf.sort_index(level=levels, sort_remaining=True)
    assert_eq(expect, got)


@pytest.mark.parametrize("idx_get", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("cols_get", [0, 1, [0, 1], [1, 0], [1], [0]])
def test_multiindex_loc_scalar(idx_get, cols_get):
    idx = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    df = cudf.DataFrame({0: range(4), 1: range(10, 50, 10)}, index=idx)
    pdf = df.to_pandas()

    actual = df.loc[idx_get, cols_get]
    expected = pdf.loc[idx_get, cols_get]

    assert_eq(actual, expected)
