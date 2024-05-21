# Copyright (c) 2019-2024, NVIDIA CORPORATION.

"""
Test related to MultiIndex
"""

import datetime
import itertools
import operator
import pickle
import re
from contextlib import contextmanager
from io import BytesIO

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.core.column import as_column
from cudf.core.index import as_index
from cudf.testing._utils import (
    assert_eq,
    assert_exceptions_equal,
    assert_neq,
    expect_warning_if,
)


@contextmanager
def expect_pandas_performance_warning(idx):
    with expect_warning_if(
        (not isinstance(idx[0], tuple) and len(idx) > 2)
        or (isinstance(idx[0], tuple) and len(idx[0]) > 2),
        pd.errors.PerformanceWarning,
    ):
        yield


def test_multiindex_levels_codes_validation():
    levels = [["a", "b"], ["c", "d"]]

    # Codes not a sequence of sequences
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [0, 1]],),
        rfunc_args_and_kwargs=([levels, [0, 1]],),
    )

    # Codes don't match levels
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [[0], [1], [1]]],),
        rfunc_args_and_kwargs=([levels, [[0], [1], [1]]],),
    )

    # Largest code greater than number of levels
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [[0, 1], [0, 2]]],),
        rfunc_args_and_kwargs=([levels, [[0, 1], [0, 2]]],),
    )

    # Unequal code lengths
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [[0, 1], [0]]],),
        rfunc_args_and_kwargs=([levels, [[0, 1], [0]]],),
    )
    # Didn't pass levels and codes
    assert_exceptions_equal(lfunc=pd.MultiIndex, rfunc=cudf.MultiIndex)

    # Didn't pass non zero levels and codes
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([[], []],),
        rfunc_args_and_kwargs=([[], []],),
    )


def test_multiindex_construction():
    levels = [["a", "b"], ["c", "d"]]
    codes = [[0, 1], [1, 0]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels=levels, codes=codes)
    assert_eq(pmi, mi)


def test_multiindex_types():
    codes = [[0, 1], [1, 0]]
    levels = [[0, 1], [2, 3]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    levels = [[1.2, 2.1], [1.3, 3.1]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    levels = [["a", "b"], ["c", "d"]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)


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


def test_multiindex_swaplevel():
    midx = cudf.MultiIndex(
        levels=[
            ["lama", "cow", "falcon"],
            ["speed", "weight", "length"],
            ["first", "second"],
        ],
        codes=[
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        names=["Col1", "Col2", "Col3"],
    )
    pd_midx = midx.to_pandas()

    assert_eq(pd_midx.swaplevel(-1, -2), midx.swaplevel(-1, -2))
    assert_eq(pd_midx.swaplevel(2, 1), midx.swaplevel(2, 1))
    assert_eq(midx.swaplevel(2, 1), midx.swaplevel(1, 2))
    assert_eq(pd_midx.swaplevel(0, 2), midx.swaplevel(0, 2))
    assert_eq(pd_midx.swaplevel(2, 0), midx.swaplevel(2, 0))
    assert_eq(midx.swaplevel(1, 1), midx.swaplevel(1, 1))


def test_string_index():
    from cudf.core.index import Index

    pdf = pd.DataFrame(np.random.rand(5, 5))
    gdf = cudf.from_pandas(pdf)
    stringIndex = ["a", "b", "c", "d", "e"]
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = np.array(["a", "b", "c", "d", "e"])
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = Index(["a", "b", "c", "d", "e"], name="name")
    pdf.index = stringIndex.to_pandas()
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = as_index(as_column(["a", "b", "c", "d", "e"]), name="name")
    pdf.index = stringIndex.to_pandas()
    gdf.index = stringIndex
    assert_eq(pdf, gdf)


def test_multiindex_row_shape():
    pdf = pd.DataFrame(np.random.rand(0, 5))
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
    return pd.DataFrame(np.random.rand(7, 5))


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
    pdf.index = pdfIndex
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf, gdf)


def test_multiindex_transpose(pdf, pdfIndex):
    pdf.index = pdfIndex
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.transpose(), gdf.transpose())


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
    ps = pd.Series(np.random.rand(7))
    gs = cudf.from_pandas(ps)
    ps.index = pdfIndex
    gs.index = cudf.from_pandas(pdfIndex)
    assert_eq(ps, gs)


def test_multiindex_take(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.index.take([0]), gdf.index.take([0]))
    assert_eq(pdf.index.take(np.array([0])), gdf.index.take(np.array([0])))
    from cudf import Series

    assert_eq(pdf.index.take(pd.Series([0])), gdf.index.take(Series([0])))
    assert_eq(pdf.index.take([0, 1]), gdf.index.take([0, 1]))
    assert_eq(
        pdf.index.take(np.array([0, 1])), gdf.index.take(np.array([0, 1]))
    )
    assert_eq(
        pdf.index.take(pd.Series([0, 1])), gdf.index.take(Series([0, 1]))
    )


def test_multiindex_getitem(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
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


@pytest.mark.parametrize(
    "indexer",
    [
        (([1, 1], [0, 1]), slice(None)),
        (([1, 1], [1, 0]), slice(None)),
    ],
)
def test_multiindex_compatible_ordering(indexer):
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
def test_multiindex_loc_slice(pdf, gdf, pdfIndex, arg):
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[arg], gdf.loc[arg])


def test_multiindex_loc_errors(pdf, gdf, pdfIndex):
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
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with pytest.warns(pd.errors.PerformanceWarning):
        expected = pdf.loc[("a", "store", "clouds", "fire"), :][0]
    got = gdf.loc[("a", "store", "clouds", "fire"), :][0]
    assert_eq(expected, got)


def test_multiindex_loc_rows_0(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
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
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    print(pdf.loc[("c", "forest"), :])
    print(gdf.loc[("c", "forest"), :].to_pandas())
    assert_eq(pdf.loc[("c", "forest"), :], gdf.loc[("c", "forest"), :])


def test_multiindex_loc_rows_1_1_key(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    print(pdf.loc[("c",), :])
    print(gdf.loc[("c",), :].to_pandas())
    assert_eq(pdf.loc[("c",), :], gdf.loc[("c",), :])


def test_multiindex_column_shape():
    pdf = pd.DataFrame(np.random.rand(5, 0))
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
def test_multiindex_columns(pdf, gdf, pdfIndex, query):
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


def test_multiindex_from_tuples():
    arrays = [["a", "a", "b", "b"], ["house", "store", "house", "store"]]
    tuples = list(zip(*arrays))
    pmi = pd.MultiIndex.from_tuples(tuples)
    gmi = cudf.MultiIndex.from_tuples(tuples)
    assert_eq(pmi, gmi)


def test_multiindex_from_dataframe():
    if not hasattr(pd.MultiIndex([[]], [[]]), "codes"):
        pytest.skip()
    pdf = pd.DataFrame(
        [["a", "house"], ["a", "store"], ["b", "house"], ["b", "store"]]
    )
    gdf = cudf.from_pandas(pdf)
    pmi = pd.MultiIndex.from_frame(pdf, names=["alpha", "location"])
    gmi = cudf.MultiIndex.from_frame(gdf, names=["alpha", "location"])
    assert_eq(pmi, gmi)


@pytest.mark.parametrize(
    "arrays",
    [
        [["a", "a", "b", "b"], ["house", "store", "house", "store"]],
        [["a", "n", "n"] * 1000, ["house", "store", "house", "store"]],
        [
            ["a", "n", "n"],
            ["house", "store", "house", "store", "store"] * 1000,
        ],
        [
            ["a", "a", "n"] * 50,
            ["house", "store", "house", "store", "store"] * 100,
        ],
    ],
)
def test_multiindex_from_product(arrays):
    pmi = pd.MultiIndex.from_product(arrays, names=["alpha", "location"])
    gmi = cudf.MultiIndex.from_product(arrays, names=["alpha", "location"])
    assert_eq(pmi, gmi)


def test_multiindex_index_and_columns():
    gdf = cudf.DataFrame()
    gdf["x"] = np.random.randint(0, 5, 5)
    gdf["y"] = np.random.randint(0, 5, 5)
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
    pdf = pd.DataFrame(
        {
            "a": [4, 17, 4, 9, 5],
            "b": [1, 4, 4, 3, 2],
            "x": np.random.normal(size=5),
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
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=1000),
            "y": np.random.randint(0, 10, size=1000),
            "z": np.random.normal(size=1000),
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
    "data",
    [
        {
            "Date": [
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
            ],
            "Close": [
                3400.00,
                3401.80,
                3450.96,
                226.58,
                228.91,
                225.53,
                505.13,
                525.91,
                534.98,
            ],
            "Symbol": [
                "AMZN",
                "AMZN",
                "AMZN",
                "MSFT",
                "MSFT",
                "MSFT",
                "NVDA",
                "NVDA",
                "NVDA",
            ],
        }
    ],
)
@pytest.mark.parametrize("names", [["X", "Y"]])
def test_multiindex_copy_sem(data, names):
    """Test semantic equality for MultiIndex.copy"""
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    gdf = gdf.groupby(["Date", "Symbol"], sort=True).mean()
    pdf = pdf.groupby(["Date", "Symbol"], sort=True).mean()

    gmi = gdf.index
    gmi_copy = gmi.copy(names=names)

    pmi = pdf.index
    pmi_copy = pmi.copy(names=names)

    for glv, plv in zip(gmi_copy.levels, pmi_copy.levels):
        assert all(glv.values_host == plv.values)
    for gval, pval in zip(gmi.codes, pmi.codes):
        assert_eq(gval, pval)
    assert_eq(gmi_copy.names, pmi_copy.names)

    # Test same behavior when used on DataFrame
    gdf.index = gmi_copy
    pdf.index = pmi_copy
    assert repr(gdf) == repr(pdf)


@pytest.mark.parametrize(
    "data",
    [
        {
            "Date": [
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
                "2020-08-27",
                "2020-08-28",
                "2020-08-31",
            ],
            "Close": [
                3400.00,
                3401.80,
                3450.96,
                226.58,
                228.91,
                225.53,
                505.13,
                525.91,
                534.98,
            ],
            "Symbol": [
                "AMZN",
                "AMZN",
                "AMZN",
                "MSFT",
                "MSFT",
                "MSFT",
                "NVDA",
                "NVDA",
                "NVDA",
            ],
        },
        cudf.MultiIndex(
            levels=[[1001, 1002], [2001, 2002]],
            codes=[[1, 1, 0, 0], [0, 1, 0, 1]],
            names=["col1", "col2"],
        ),
    ],
)
@pytest.mark.parametrize("copy_on_write", [True, False])
@pytest.mark.parametrize("deep", [True, False])
def test_multiindex_copy_deep(data, copy_on_write, deep):
    """Test memory identity for deep copy
    Case1: Constructed from GroupBy, StringColumns
    Case2: Constructed from MultiIndex, NumericColumns
    """
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", copy_on_write)

    if isinstance(data, dict):
        import operator
        from functools import reduce

        gdf = cudf.DataFrame(data)
        mi1 = gdf.groupby(["Date", "Symbol"]).mean().index
        mi2 = mi1.copy(deep=deep)

        lchildren = [col.children for _, col in mi1._data.items()]
        rchildren = [col.children for _, col in mi2._data.items()]

        # Flatten
        lchildren = reduce(operator.add, lchildren)
        rchildren = reduce(operator.add, rchildren)

        lptrs = [child.base_data.get_ptr(mode="read") for child in lchildren]
        rptrs = [child.base_data.get_ptr(mode="read") for child in rchildren]

        assert all((x == y) for x, y in zip(lptrs, rptrs))

    elif isinstance(data, cudf.MultiIndex):
        same_ref = (not deep) or (
            cudf.get_option("copy_on_write") and not deep
        )
        mi1 = data
        mi2 = mi1.copy(deep=deep)

        # Assert ._levels identity
        lptrs = [
            lv._data._data[None].base_data.get_ptr(mode="read")
            for lv in mi1._levels
        ]
        rptrs = [
            lv._data._data[None].base_data.get_ptr(mode="read")
            for lv in mi2._levels
        ]

        assert all((x == y) == same_ref for x, y in zip(lptrs, rptrs))

        # Assert ._codes identity
        lptrs = [
            c.base_data.get_ptr(mode="read")
            for _, c in mi1._codes._data.items()
        ]
        rptrs = [
            c.base_data.get_ptr(mode="read")
            for _, c in mi2._codes._data.items()
        ]

        assert all((x == y) == same_ref for x, y in zip(lptrs, rptrs))

        # Assert ._data identity
        lptrs = [
            d.base_data.get_ptr(mode="read") for _, d in mi1._data.items()
        ]
        rptrs = [
            d.base_data.get_ptr(mode="read") for _, d in mi2._data.items()
        ]

        assert all((x == y) == same_ref for x, y in zip(lptrs, rptrs))
    cudf.set_option("copy_on_write", original_cow_setting)


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
    tuples = list(zip(*arrays))
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


def test_multiindex_to_frame(pdfIndex, pdfIndexNulls):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.to_frame(), gdfIndex.to_frame())

    gdfIndex = cudf.from_pandas(pdfIndexNulls)
    assert_eq(
        pdfIndexNulls.to_frame().fillna("nan"),
        gdfIndex.to_frame().fillna("nan"),
    )


def test_multiindex_groupby_to_frame():
    gdf = cudf.DataFrame(
        {"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5], "z": [0, 1, 0, 1, 0]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["x", "y"], sort=True).count()
    pdg = pdf.groupby(["x", "y"], sort=True).count()
    assert_eq(pdg.index.to_frame(), gdg.index.to_frame())


def test_multiindex_reset_index(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
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
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(gdf, pdf)
    assert_eq(gdf.T, pdf.T)


def test_multiindex_rows_with_wildcard(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
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
    pdf = pdf.T
    pdf.columns = pdfIndex
    gdf = cudf.from_pandas(pdf)
    pdf["d"] = [1, 2, 3, 4, 5]
    gdf["d"] = [1, 2, 3, 4, 5]
    assert_eq(pdf, gdf)


def test_multiindex_iter_error():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{midx.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(midx)


def test_multiindex_values():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )

    result = midx.values

    assert isinstance(result, cp.ndarray)
    np.testing.assert_array_equal(
        result.get(), np.array([[1, 1], [1, 5], [3, 2], [4, 2], [5, 1]])
    )


def test_multiindex_values_host():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    pmidx = midx.to_pandas()

    assert_eq(midx.values_host, pmidx.values)


def test_multiindex_to_numpy():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    pmidx = midx.to_pandas()

    assert_eq(midx.to_numpy(), pmidx.to_numpy())


@pytest.mark.parametrize(
    "gdi, fill_value, expected",
    [
        (
            cudf.MultiIndex(
                levels=[[1, 3, 4, None], [1, 2, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            5,
            cudf.MultiIndex(
                levels=[[1, 3, 4, 5], [1, 2, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
        ),
        (
            cudf.MultiIndex(
                levels=[[1, 3, 4, None], [1, None, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            100,
            cudf.MultiIndex(
                levels=[[1, 3, 4, 100], [1, 100, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
        ),
        (
            cudf.MultiIndex(
                levels=[["a", "b", "c", None], ["1", None, "5"]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            "100",
            cudf.MultiIndex(
                levels=[["a", "b", "c", "100"], ["1", "100", "5"]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
        ),
    ],
)
def test_multiindex_fillna(gdi, fill_value, expected):
    assert_eq(expected, gdi.fillna(fill_value))


@pytest.mark.parametrize(
    "pdi",
    [
        pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=["one", "two", "three"],
        ),
        pd.MultiIndex.from_tuples(
            list(
                zip(
                    *[
                        [
                            "bar",
                            "bar",
                            "baz",
                            "baz",
                            "foo",
                            "foo",
                            "qux",
                            "qux",
                        ],
                        [
                            "one",
                            "two",
                            "one",
                            "two",
                            "one",
                            "two",
                            "one",
                            "two",
                        ],
                    ]
                )
            )
        ),
    ],
)
def test_multiindex_empty(pdi):
    gdi = cudf.from_pandas(pdi)

    assert_eq(pdi.empty, gdi.empty)


@pytest.mark.parametrize(
    "pdi",
    [
        pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=["one", "two", "three"],
        ),
        pd.MultiIndex.from_tuples(
            list(
                zip(
                    *[
                        [
                            "bar",
                            "bar",
                            "baz",
                            "baz",
                            "foo",
                            "foo",
                            "qux",
                            "qux",
                        ],
                        [
                            "one",
                            "two",
                            "one",
                            "two",
                            "one",
                            "two",
                            "one",
                            "two",
                        ],
                    ]
                )
            )
        ),
    ],
)
def test_multiindex_size(pdi):
    gdi = cudf.from_pandas(pdi)

    assert_eq(pdi.size, gdi.size)


@pytest.mark.parametrize(
    "level",
    [
        [],
        "alpha",
        "location",
        "weather",
        0,
        1,
        [0, 1],
        -1,
        [-1, -2],
        [-1, "weather"],
    ],
)
def test_multiindex_droplevel_simple(pdfIndex, level):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.droplevel(level), gdfIndex.droplevel(level))


@pytest.mark.parametrize(
    "level",
    itertools.chain(
        *(
            itertools.combinations(
                ("alpha", "location", "weather", "sign", "timestamp"), r
            )
            for r in range(5)
        )
    ),
)
def test_multiindex_droplevel_name(pdfIndex, level):
    level = list(level)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.droplevel(level), gdfIndex.droplevel(level))


@pytest.mark.parametrize(
    "level",
    itertools.chain(*(itertools.combinations(range(5), r) for r in range(5))),
)
def test_multiindex_droplevel_index(pdfIndex, level):
    level = list(level)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.droplevel(level), gdfIndex.droplevel(level))


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("return_indexer", [True, False])
@pytest.mark.parametrize(
    "pmidx",
    [
        pd.MultiIndex(
            levels=[[1, 3, 4, 5], [1, 2, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex.from_product(
            [["bar", "baz", "foo", "qux"], ["one", "two"]],
            names=["first", "second"],
        ),
        pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=["one", "two", "three"],
        ),
        pd.MultiIndex.from_tuples(
            list(
                zip(
                    *[
                        [
                            "bar",
                            "bar",
                            "baz",
                            "baz",
                            "foo",
                            "foo",
                            "qux",
                            "qux",
                        ],
                        [
                            "one",
                            "two",
                            "one",
                            "two",
                            "one",
                            "two",
                            "one",
                            "two",
                        ],
                    ]
                )
            )
        ),
    ],
)
def test_multiindex_sort_values(pmidx, ascending, return_indexer):
    pmidx = pmidx
    midx = cudf.from_pandas(pmidx)

    expected = pmidx.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )
    actual = midx.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )

    if return_indexer:
        expected_indexer = expected[1]
        actual_indexer = actual[1]

        assert_eq(expected_indexer, actual_indexer)

        expected = expected[0]
        actual = actual[0]

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pdi",
    [
        pd.MultiIndex(
            levels=[[1, 3.0, 4, 5], [1, 2.3, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex(
            levels=[[1, 3, 4, -10], [1, 11, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex(
            levels=[["a", "b", "c", "100"], ["1", "100", "5"]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pytest.param(
            pd.MultiIndex(
                levels=[[None, "b", "c", "a"], ["1", None, "5"]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            marks=[
                pytest.mark.xfail(
                    reason="https://github.com/pandas-dev/pandas/issues/35584"
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
def test_multiindex_argsort(pdi, ascending):
    gdi = cudf.from_pandas(pdi)

    if not ascending:
        expected = pdi.argsort()[::-1]
    else:
        expected = pdi.argsort()

    actual = gdi.argsort(ascending=ascending)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx", [pd.MultiIndex.from_product([["python", "cobra"], [2018, 2019]])]
)
@pytest.mark.parametrize(
    "names", [[None, None], ["a", None], ["new name", "another name"]]
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_set_names(idx, names, inplace):
    pi = idx.copy()
    gi = cudf.from_pandas(idx)

    expected = pi.set_names(names=names, inplace=inplace)
    actual = gi.set_names(names=names, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_product(
            [["python", "cobra"], [2018, 2019], ["aab", "bcd"]]
        ),
        pd.MultiIndex.from_product(
            [["python", "cobra"], [2018, 2019], ["aab", "bcd"]],
            names=[1, 0, 2],
        ),
    ],
)
@pytest.mark.parametrize(
    "level, names",
    [
        (0, "abc"),
        (1, "xyz"),
        ([2, 1], ["a", "b"]),
        ([0, 1], ["aa", "bb"]),
        (None, ["a", "b", "c"]),
        (None, ["a", None, "c"]),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_set_names_default_and_int_names(
    idx, level, names, inplace
):
    pi = idx.copy()
    gi = cudf.from_pandas(idx)

    expected = pi.set_names(names=names, level=level, inplace=inplace)
    actual = gi.set_names(names=names, level=level, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_product(
            [["python", "cobra"], [2018, 2019], ["aab", "bcd"]],
            names=["one", None, "three"],
        ),
    ],
)
@pytest.mark.parametrize(
    "level, names",
    [
        ([None], "abc"),
        (["three", "one"], ["a", "b"]),
        (["three", 1], ["a", "b"]),
        ([0, "three", 1], ["a", "b", "z"]),
        (["one", 1, "three"], ["a", "b", "z"]),
        (["one", None, "three"], ["a", "b", "z"]),
        ([2, 1], ["a", "b"]),
        (1, "xyz"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_set_names_string_names(idx, level, names, inplace):
    pi = idx.copy()
    gi = cudf.from_pandas(idx)

    expected = pi.set_names(names=names, level=level, inplace=inplace)
    actual = gi.set_names(names=names, level=level, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "level, names", [(1, ["a"]), (None, "a"), ([1, 2], ["a"]), (None, ["a"])]
)
def test_multiindex_set_names_error(level, names):
    pi = pd.MultiIndex.from_product(
        [["python", "cobra"], [2018, 2019], ["aab", "bcd"]]
    )
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        lfunc=pi.set_names,
        rfunc=gi.set_names,
        lfunc_args_and_kwargs=([], {"names": names, "level": level}),
        rfunc_args_and_kwargs=([], {"names": names, "level": level}),
    )


@pytest.mark.parametrize(
    "idx",
    [
        pd.MultiIndex.from_product([["python", "cobra"], [2018, 2019]]),
        pd.MultiIndex.from_product(
            [["python", "cobra"], [2018, 2019]], names=["old name", None]
        ),
    ],
)
@pytest.mark.parametrize(
    "names",
    [
        [None, None],
        ["a", None],
        ["new name", "another name"],
        [1, None],
        [2, 3],
        [42, "name"],
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_rename(idx, names, inplace):
    pi = idx.copy()
    gi = cudf.from_pandas(idx)

    expected = pi.rename(names=names, inplace=inplace)
    actual = gi.rename(names=names, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "names", ["plain string", 123, ["str"], ["l1", "l2", "l3"]]
)
def test_multiindex_rename_error(names):
    pi = pd.MultiIndex.from_product([["python", "cobra"], [2018, 2019]])
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        lfunc=pi.rename,
        rfunc=gi.rename,
        lfunc_args_and_kwargs=([], {"names": names}),
        rfunc_args_and_kwargs=([], {"names": names}),
    )


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


def test_multiindex_duplicate_names():
    gi = cudf.MultiIndex(
        levels=[["a", "b"], ["b", "a"]],
        codes=[[0, 0], [0, 1]],
        names=["a", "a"],
    )
    pi = pd.MultiIndex(
        levels=[["a", "b"], ["b", "a"]],
        codes=[[0, 0], [0, 1]],
        names=["a", "a"],
    )

    assert_eq(gi, pi)


def test_difference():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    midx2 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3, 3], [0, 2, 1, 1, 0, 2]],
        names=["x", "y"],
    )

    expected = midx2.to_pandas().difference(midx.to_pandas())
    actual = midx2.difference(midx)
    assert isinstance(actual, cudf.MultiIndex)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (
            pd.MultiIndex.from_arrays(
                [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], ["Red", "Blue", "Red", "Blue"]],
                names=["a", "b"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], ["Red", "Green", "Red", "Green"]],
                names=["x", "y"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], [0.2, 0.4, 1.4, 10], [3, 3, 2, 4]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            [(2, 6, 12)],
        ),
    ],
)
@pytest.mark.parametrize("sort", [None, False])
def test_union_mulitIndex(idx1, idx2, sort):
    expected = idx1.union(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.MultiIndex) else idx1
    idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.MultiIndex) else idx2

    actual = idx1.union(idx2, sort=sort)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (
            pd.MultiIndex.from_arrays(
                [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
            ),
            pd.MultiIndex.from_arrays(
                [[1, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], ["Red", "Blue", "Red", "Blue"]],
                names=["a", "b"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], ["Red", "Green", "Red", "Green"]],
                names=["x", "y"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], [0.2, 0.4, 1.4, 10], [3, 3, 2, 4]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
            ),
        ),
    ],
)
@pytest.mark.parametrize("sort", [None, False])
def test_intersection_mulitIndex(idx1, idx2, sort):
    expected = idx1.intersection(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1)
    idx2 = cudf.from_pandas(idx2)

    actual = idx1.intersection(idx2, sort=sort)
    assert_eq(expected, actual, exact=False)


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


@pytest.mark.parametrize(
    "pidx",
    [
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
            names=["a", "b", "c"],
        ),
        pd.MultiIndex.from_arrays(
            [[1.0, 2, 3, 4], [5, 6, 7.8, 10], [11, 12, 12, 13]],
        ),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        "is_numeric",
        "is_boolean",
        "is_integer",
        "is_floating",
        "is_object",
        "is_categorical",
        "is_interval",
    ],
)
def test_multiindex_type_methods(pidx, func):
    gidx = cudf.from_pandas(pidx)

    with pytest.warns(FutureWarning):
        expected = getattr(pidx, func)()

    with pytest.warns(FutureWarning):
        actual = getattr(gidx, func)()

    if func == "is_object":
        assert_eq(False, actual)
    else:
        assert_eq(expected, actual)


def test_multiindex_index_single_row():
    arrays = [["a", "a", "b", "b"], [1, 2, 3, 4]]
    tuples = list(zip(*arrays))
    idx = cudf.MultiIndex.from_tuples(tuples)
    gdf = cudf.DataFrame(
        {"first": cp.random.rand(4), "second": cp.random.rand(4)}
    )
    gdf.index = idx
    pdf = gdf.to_pandas()
    assert_eq(pdf.loc[("b", 3)], gdf.loc[("b", 3)])


def test_multiindex_levels():
    gidx = cudf.MultiIndex.from_product(
        [range(3), ["one", "two"]], names=["first", "second"]
    )
    pidx = gidx.to_pandas()

    assert_eq(gidx.levels[0], pidx.levels[0])
    assert_eq(gidx.levels[1], pidx.levels[1])


def test_multiindex_empty_slice_pandas_compatibility():
    expected = pd.MultiIndex.from_tuples([("a", "b")])[:0]
    with cudf.option_context("mode.pandas_compatible", True):
        actual = cudf.from_pandas(expected)
    assert_eq(expected, actual, exact=False)


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


def test_multiindex_to_series_error():
    midx = cudf.MultiIndex.from_tuples([("a", "b")])
    with pytest.raises(NotImplementedError):
        midx.to_series()


@pytest.mark.parametrize(
    "pidx",
    [
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
            names=["a", "b", "c"],
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
            names=["a", "a", "a"],
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
        ),
    ],
)
@pytest.mark.parametrize(
    "name", [None, no_default, ["x", "y", "z"], ["rapids", "rapids", "rapids"]]
)
@pytest.mark.parametrize("allow_duplicates", [True, False])
@pytest.mark.parametrize("index", [True, False])
def test_multiindex_to_frame_allow_duplicates(
    pidx, name, allow_duplicates, index
):
    gidx = cudf.from_pandas(pidx)

    if name is None or (
        (
            len(pidx.names) != len(set(pidx.names))
            and not all(x is None for x in pidx.names)
        )
        and not allow_duplicates
        and name is no_default
    ):
        assert_exceptions_equal(
            pidx.to_frame,
            gidx.to_frame,
            lfunc_args_and_kwargs=(
                [],
                {
                    "index": index,
                    "name": name,
                    "allow_duplicates": allow_duplicates,
                },
            ),
            rfunc_args_and_kwargs=(
                [],
                {
                    "index": index,
                    "name": name,
                    "allow_duplicates": allow_duplicates,
                },
            ),
        )
    else:
        if (
            len(pidx.names) != len(set(pidx.names))
            and not all(x is None for x in pidx.names)
            and not isinstance(name, list)
        ) or (isinstance(name, list) and len(name) != len(set(name))):
            # cudf doesn't have the ability to construct dataframes
            # with duplicate column names
            with pytest.raises(ValueError):
                gidx.to_frame(
                    index=index,
                    name=name,
                    allow_duplicates=allow_duplicates,
                )
        else:
            expected = pidx.to_frame(
                index=index, name=name, allow_duplicates=allow_duplicates
            )
            actual = gidx.to_frame(
                index=index, name=name, allow_duplicates=allow_duplicates
            )

            assert_eq(expected, actual)


@pytest.mark.parametrize("bad", ["foo", ["foo"]])
def test_multiindex_set_names_validation(bad):
    mi = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    with pytest.raises(ValueError):
        mi.names = bad


def test_multiindex_values_pandas_compatible():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            midx.values


def test_multiindex_dtype_error():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    with pytest.raises(TypeError):
        cudf.Index(midx, dtype="int64")
    with pytest.raises(TypeError):
        cudf.Index(midx.to_pandas(), dtype="int64")


def test_multiindex_codes():
    midx = cudf.MultiIndex.from_tuples(
        [("a", "b"), ("a", "c"), ("b", "c")], names=["A", "Z"]
    )

    for p_array, g_array in zip(midx.to_pandas().codes, midx.codes):
        assert_eq(p_array, g_array)


def test_multiindex_union_error():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    pidx = midx.to_pandas()

    assert_exceptions_equal(
        midx.union,
        pidx.union,
        lfunc_args_and_kwargs=(["a"],),
        rfunc_args_and_kwargs=(["b"],),
    )


@pytest.mark.parametrize("idx_get", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("cols_get", [0, 1, [0, 1], [1, 0], [1], [0]])
def test_multiindex_loc_scalar(idx_get, cols_get):
    idx = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    df = cudf.DataFrame({0: range(4), 1: range(10, 50, 10)}, index=idx)
    pdf = df.to_pandas()

    actual = df.loc[idx_get, cols_get]
    expected = pdf.loc[idx_get, cols_get]

    assert_eq(actual, expected)


def test_multiindex_eq_other_multiindex():
    idx = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    result = idx == idx
    expected = np.array([True, True])
    assert_eq(result, expected)


@pytest.fixture(
    params=[
        "from_product",
        "from_tuples",
        "from_arrays",
        "init",
    ]
)
def midx(request):
    if request.param == "from_product":
        return cudf.MultiIndex.from_product([[0, 1], [1, 0]])
    elif request.param == "from_tuples":
        return cudf.MultiIndex.from_tuples([(0, 1), (0, 0), (1, 1), (1, 0)])
    elif request.param == "from_arrays":
        return cudf.MultiIndex.from_arrays([[0, 0, 1, 1], [1, 0, 1, 0]])
    elif request.param == "init":
        return cudf.MultiIndex(
            levels=[[0, 1], [0, 1]], codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        )
    else:
        raise NotImplementedError(f"{request.param} not implemented")


def test_multindex_constructor_levels_always_indexes(midx):
    assert_eq(midx.levels[0], cudf.Index([0, 1]))
    assert_eq(midx.levels[1], cudf.Index([0, 1]))


@pytest.mark.parametrize(
    "array",
    [
        list,
        tuple,
        np.array,
        cp.array,
        pd.Index,
        cudf.Index,
        pd.Series,
        cudf.Series,
    ],
)
def test_multiindex_from_arrays(array):
    pd_data = [[0, 0, 1, 1], [1, 0, 1, 0]]
    cudf_data = [array(lst) for lst in pd_data]
    result = pd.MultiIndex.from_arrays(pd_data)
    expected = cudf.MultiIndex.from_arrays(cudf_data)
    assert_eq(result, expected)


@pytest.mark.parametrize("arg", ["foo", ["foo"]])
def test_multiindex_from_arrays_wrong_arg(arg):
    with pytest.raises(TypeError):
        cudf.MultiIndex.from_arrays(arg)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
    ],
)
def test_index_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = pa.array([scalar, None])
    midx = cudf.MultiIndex(levels=[pa_array], codes=[[0]])
    with pytest.raises(ValueError):
        midx.to_pandas(nullable=True, arrow_type=True)


@pytest.mark.parametrize(
    "scalar",
    [1, 1.0, "a", datetime.datetime(2020, 1, 1), datetime.timedelta(1)],
)
def test_index_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    midx = cudf.MultiIndex(levels=[pa_array], codes=[[0]])
    result = midx.to_pandas(arrow_type=True)
    expected = pd.MultiIndex(
        levels=[pd.arrays.ArrowExtensionArray(pa_array)], codes=[[0]]
    )
    pd.testing.assert_index_equal(result, expected)


def test_multi_index_contains_hashable():
    gidx = cudf.MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3]))
    pidx = gidx.to_pandas()

    assert_exceptions_equal(
        lambda: [] in gidx,
        lambda: [] in pidx,
        lfunc_args_and_kwargs=((),),
        rfunc_args_and_kwargs=((),),
    )
