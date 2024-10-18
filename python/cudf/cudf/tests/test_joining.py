# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from itertools import combinations, product, repeat

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.core.dtypes import CategoricalDtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import assert_eq
from cudf.testing._utils import (
    INTEGER_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
)

_JOIN_TYPES = ("left", "inner", "outer", "right", "leftanti", "leftsemi")


def make_params():
    rng = np.random.default_rng(seed=0)

    hows = _JOIN_TYPES

    # Test specific cases (1)
    aa = [0, 0, 4, 5, 5]
    bb = [0, 0, 2, 3, 5]
    for how in hows:
        yield (aa, bb, how)

    # Test specific cases (2)
    aa = [0, 0, 1, 2, 3]
    bb = [0, 1, 2, 2, 3]
    for how in hows:
        yield (aa, bb, how)

    # Test large random integer inputs
    aa = rng.integers(0, 50, 100)
    bb = rng.integers(0, 50, 100)
    for how in hows:
        yield (aa, bb, how)

    # Test floating point inputs
    aa = rng.random(50)
    bb = rng.random(50)
    for how in hows:
        yield (aa, bb, how)


def pd_odd_joins(left, right, join_type):
    if join_type == "leftanti":
        return left[~left.index.isin(right.index)][left.columns]
    elif join_type == "leftsemi":
        return left[left.index.isin(right.index)][left.columns]


def assert_join_results_equal(expect, got, how, **kwargs):
    if how not in _JOIN_TYPES:
        raise ValueError(f"Unrecognized join type {how}")
    if how == "right":
        got = got[expect.columns]

    if isinstance(expect, (pd.Series, cudf.Series)):
        return assert_eq(
            expect.sort_values().reset_index(drop=True),
            got.sort_values().reset_index(drop=True),
            **kwargs,
        )
    elif isinstance(expect, (pd.DataFrame, cudf.DataFrame)):
        if not len(
            expect.columns
        ):  # can't sort_values() on a df without columns
            return assert_eq(expect, got, **kwargs)

        assert_eq(
            expect.sort_values(expect.columns.to_list()).reset_index(
                drop=True
            ),
            got.sort_values(got.columns.to_list()).reset_index(drop=True),
            **kwargs,
        )
    elif isinstance(expect, (pd.Index, cudf.Index)):
        return assert_eq(expect.sort_values(), got.sort_values(), **kwargs)
    else:
        raise ValueError(f"Not a join result: {type(expect).__name__}")


@pytest.mark.parametrize("aa,bb,how", make_params())
def test_dataframe_join_how(aa, bb, how):
    df = cudf.DataFrame()
    df["a"] = aa
    df["b"] = bb

    def work_pandas(df, how):
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        if how == "leftanti":
            joined = pd_odd_joins(df1, df2, "leftanti")
        elif how == "leftsemi":
            joined = pd_odd_joins(df1, df2, "leftsemi")
        else:
            joined = df1.join(df2, how=how, sort=True)
        return joined

    def work_gdf(df):
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        joined = df1.join(df2, how=how, sort=True)
        return joined

    expect = work_pandas(df.to_pandas(), how)
    got = work_gdf(df)
    expecto = expect.copy()
    goto = got.copy()

    expect = expect.astype(np.float64).fillna(np.nan)[expect.columns]
    got = got.astype(np.float64).fillna(np.nan)[expect.columns]

    assert got.index.name is None

    assert list(expect.columns) == list(got.columns)
    if how in {"left", "inner", "right", "leftanti", "leftsemi"}:
        assert_eq(sorted(expect.index.values), sorted(got.index.values))
        if how != "outer":
            # Newly introduced ambiguous ValueError thrown when
            # an index and column have the same name. Rename the
            # index so sorts work.
            # TODO: What is the less hacky way?
            expect.index.name = "bob"
            got.index.name = "mary"
            assert_join_results_equal(expect, got, how=how)
        # if(how=='right'):
        #     _sorted_check_series(expect['a'], expect['b'],
        #                          got['a'], got['b'])
        # else:
        #     _sorted_check_series(expect['b'], expect['a'], got['b'],
        #                          got['a'])
        else:
            for c in expecto.columns:
                _check_series(expecto[c].fillna(-1), goto[c].fillna(-1))


def _check_series(expect, got):
    magic = 0xDEADBEAF

    direct_equal = np.all(expect.values == got.to_numpy())
    nanfilled_equal = np.all(
        expect.fillna(magic).values == got.fillna(magic).to_numpy()
    )
    msg = "direct_equal={}, nanfilled_equal={}".format(
        direct_equal, nanfilled_equal
    )
    assert direct_equal or nanfilled_equal, msg


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="bug in older version of pandas",
)
def test_dataframe_join_suffix():
    rng = np.random.default_rng(seed=0)

    df = cudf.DataFrame(rng.integers(0, 5, (5, 3)), columns=list("abc"))

    left = df.set_index("a")
    right = df.set_index("c")
    msg = (
        "there are overlapping columns but lsuffix and rsuffix are not defined"
    )
    with pytest.raises(ValueError, match=msg):
        left.join(right)

    got = left.join(right, lsuffix="_left", rsuffix="_right", sort=True)
    expect = left.to_pandas().join(
        right.to_pandas(),
        lsuffix="_left",
        rsuffix="_right",
        sort=True,
    )
    # TODO: Retain result index name
    expect.index.name = None
    assert_eq(got, expect)

    got_sorted = got.sort_values(by=["b_left", "c", "b_right"], axis=0)
    expect_sorted = expect.sort_values(by=["b_left", "c", "b_right"], axis=0)
    assert_eq(got_sorted, expect_sorted)


def test_dataframe_join_cats():
    lhs = cudf.DataFrame()
    lhs["a"] = pd.Categorical(list("aababcabbc"), categories=list("abc"))
    lhs["b"] = bb = np.arange(len(lhs))
    lhs = lhs.set_index("a")

    rhs = cudf.DataFrame()
    rhs["a"] = pd.Categorical(list("abcac"), categories=list("abc"))
    rhs["c"] = cc = np.arange(len(rhs))
    rhs = rhs.set_index("a")

    got = lhs.join(rhs)
    expect = lhs.to_pandas().join(rhs.to_pandas())

    # Note: pandas make an object Index after joining
    assert_join_results_equal(expect, got, how="inner")

    # Just do some rough checking here.
    assert list(got.columns) == ["b", "c"]
    assert len(got) > 0
    assert set(got.index.to_pandas()) & set("abc")
    assert set(got["b"].to_numpy()) & set(bb)
    assert set(got["c"].to_numpy()) & set(cc)


def test_dataframe_join_combine_cats():
    lhs = cudf.DataFrame({"join_index": ["a", "b", "c"], "data_x": [1, 2, 3]})
    rhs = cudf.DataFrame({"join_index": ["b", "c", "d"], "data_y": [2, 3, 4]})

    lhs["join_index"] = lhs["join_index"].astype("category")
    rhs["join_index"] = rhs["join_index"].astype("category")

    lhs = lhs.set_index("join_index")
    rhs = rhs.set_index("join_index")

    lhs_pd = lhs.to_pandas()
    rhs_pd = rhs.to_pandas()

    lhs_pd.index = lhs_pd.index.astype("object")
    rhs_pd.index = rhs_pd.index.astype("object")

    expect = lhs_pd.join(rhs_pd, how="outer")
    expect.index = expect.index.astype("category")
    got = lhs.join(rhs, how="outer")

    assert_eq(expect.index.sort_values(), got.index.sort_values())


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_dataframe_join_mismatch_cats(how):
    pdf1 = pd.DataFrame(
        {
            "join_col": ["a", "b", "c", "d", "e"],
            "data_col_left": [10, 20, 30, 40, 50],
        }
    )
    pdf2 = pd.DataFrame(
        {"join_col": ["c", "e", "f"], "data_col_right": [6, 7, 8]}
    )

    pdf1["join_col"] = pdf1["join_col"].astype("category")
    pdf2["join_col"] = pdf2["join_col"].astype("category")

    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    gdf1 = gdf1.set_index("join_col")
    gdf2 = gdf2.set_index("join_col")

    pdf1 = pdf1.set_index("join_col")
    pdf2 = pdf2.set_index("join_col")
    join_gdf = gdf1.join(gdf2, how=how, sort=True)
    join_pdf = pdf1.join(pdf2, how=how)

    got = join_gdf.fillna(-1).to_pandas()
    expect = join_pdf.fillna(-1)  # note: cudf join doesn't mask NA

    # We yield a categorical here whereas pandas gives Object.
    expect.index = expect.index.astype("category")
    # cudf creates the columns in different order than pandas for right join
    if how == "right":
        got = got[["data_col_left", "data_col_right"]]

    expect.data_col_right = expect.data_col_right.astype(np.int64)
    expect.data_col_left = expect.data_col_left.astype(np.int64)

    assert_join_results_equal(expect, got, how=how, check_categorical=False)


@pytest.mark.parametrize("on", ["key1", ["key1", "key2"], None])
def test_dataframe_merge_on(on):
    rng = np.random.default_rng(seed=0)

    # Make cuDF
    df_left = cudf.DataFrame()
    nelem = 500
    df_left["key1"] = rng.integers(0, 40, nelem)
    df_left["key2"] = rng.integers(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = cudf.DataFrame()
    nelem = 500
    df_right["key1"] = rng.integers(0, 30, nelem)
    df_right["key2"] = rng.integers(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    # Make pandas DF
    pddf_left = df_left.to_pandas()
    pddf_right = df_right.to_pandas()

    # Expected result (from pandas)
    pddf_joined = pddf_left.merge(pddf_right, on=on, how="left")

    # Test (from cuDF; doesn't check for ordering)
    join_result = df_left.merge(df_right, on=on, how="left")
    join_result_cudf = cudf.merge(df_left, df_right, on=on, how="left")

    join_result["right_val"] = (
        join_result["right_val"].astype(np.float64).fillna(np.nan)
    )

    join_result_cudf["right_val"] = (
        join_result_cudf["right_val"].astype(np.float64).fillna(np.nan)
    )

    for col in list(pddf_joined.columns):
        if col.count("_y") > 0:
            join_result[col] = (
                join_result[col].astype(np.float64).fillna(np.nan)
            )
            join_result_cudf[col] = (
                join_result_cudf[col].astype(np.float64).fillna(np.nan)
            )

    # Test dataframe equality (ignore order of rows and columns)
    cdf_result = (
        join_result.to_pandas()
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    pdf_result = pddf_joined.sort_values(
        list(pddf_joined.columns)
    ).reset_index(drop=True)

    assert_join_results_equal(cdf_result, pdf_result, how="left")

    merge_func_result_cdf = (
        join_result_cudf.to_pandas()
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    assert_join_results_equal(merge_func_result_cdf, cdf_result, how="left")


def test_dataframe_merge_on_unknown_column():
    rng = np.random.default_rng(seed=0)

    # Make cuDF
    df_left = cudf.DataFrame()
    nelem = 500
    df_left["key1"] = rng.integers(0, 40, nelem)
    df_left["key2"] = rng.integers(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = cudf.DataFrame()
    nelem = 500
    df_right["key1"] = rng.integers(0, 30, nelem)
    df_right["key2"] = rng.integers(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    with pytest.raises(KeyError) as raises:
        df_left.merge(df_right, on="bad_key", how="left")
    raises.match("bad_key")


def test_dataframe_merge_no_common_column():
    rng = np.random.default_rng(seed=0)

    # Make cuDF
    df_left = cudf.DataFrame()
    nelem = 500
    df_left["key1"] = rng.integers(0, 40, nelem)
    df_left["key2"] = rng.integers(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = cudf.DataFrame()
    nelem = 500
    df_right["key3"] = rng.integers(0, 30, nelem)
    df_right["key4"] = rng.integers(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    with pytest.raises(ValueError) as raises:
        df_left.merge(df_right, how="left")
    raises.match("No common columns to perform merge on")


def test_dataframe_empty_merge():
    gdf1 = cudf.DataFrame({"a": [], "b": []})
    gdf2 = cudf.DataFrame({"a": [], "c": []})

    expect = cudf.DataFrame({"a": [], "b": [], "c": []})
    got = gdf1.merge(gdf2, how="left", on=["a"])

    assert_join_results_equal(expect, got, how="left")


def test_dataframe_merge_order():
    gdf1 = cudf.DataFrame()
    gdf2 = cudf.DataFrame()
    gdf1["id"] = [10, 11]
    gdf1["timestamp"] = [1, 2]
    gdf1["a"] = [3, 4]

    gdf2["id"] = [4, 5]
    gdf2["a"] = [7, 8]

    gdf = gdf1.merge(gdf2, how="left", on=["id", "a"])

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1["id"] = [10, 11]
    df1["timestamp"] = [1, 2]
    df1["a"] = [3, 4]

    df2["id"] = [4, 5]
    df2["a"] = [7, 8]

    df = df1.merge(df2, how="left", on=["id", "a"])
    assert_join_results_equal(df, gdf, how="left")


@pytest.mark.parametrize(
    "pairs",
    [
        ("", ""),
        ("", "a"),
        ("", "ab"),
        ("", "abc"),
        ("", "b"),
        ("", "bcd"),
        ("", "cde"),
        ("a", "a"),
        ("a", "ab"),
        ("a", "abc"),
        ("a", "b"),
        ("a", "bcd"),
        ("a", "cde"),
        ("ab", "ab"),
        ("ab", "abc"),
        ("ab", "b"),
        ("ab", "bcd"),
        ("ab", "cde"),
        ("abc", "abc"),
        ("abc", "b"),
        ("abc", "bcd"),
        ("abc", "cde"),
        ("b", "b"),
        ("b", "bcd"),
        ("b", "cde"),
        ("bcd", "bcd"),
        ("bcd", "cde"),
        ("cde", "cde"),
    ],
)
@pytest.mark.parametrize("max", [5, 1000])
@pytest.mark.parametrize("rows", [1, 5, 100])
@pytest.mark.parametrize("how", ["left", "inner", "outer"])
def test_dataframe_pairs_of_triples(pairs, max, rows, how):
    rng = np.random.default_rng(seed=0)

    pdf_left = pd.DataFrame()
    pdf_right = pd.DataFrame()
    for left_column in pairs[0]:
        pdf_left[left_column] = rng.integers(0, max, rows)
    for right_column in pairs[1]:
        pdf_right[right_column] = rng.integers(0, max, rows)
    gdf_left = cudf.from_pandas(pdf_left)
    gdf_right = cudf.from_pandas(pdf_right)
    if not set(pdf_left.columns).intersection(pdf_right.columns):
        with pytest.raises(
            pd.errors.MergeError,
            match="No common columns to perform merge on",
        ):
            pdf_left.merge(pdf_right)
        with pytest.raises(
            ValueError, match="No common columns to perform merge on"
        ):
            gdf_left.merge(gdf_right)
    elif not [value for value in pdf_left if value in pdf_right]:
        with pytest.raises(
            pd.errors.MergeError,
            match="No common columns to perform merge on",
        ):
            pdf_left.merge(pdf_right)
        with pytest.raises(
            ValueError, match="No common columns to perform merge on"
        ):
            gdf_left.merge(gdf_right)
    else:
        pdf_result = pdf_left.merge(pdf_right, how=how)
        gdf_result = gdf_left.merge(gdf_right, how=how)
        assert np.array_equal(gdf_result.columns, pdf_result.columns)
        for column in gdf_result:
            gdf_col_result_sorted = gdf_result[column].fillna(-1).sort_values()
            pd_col_result_sorted = pdf_result[column].fillna(-1).sort_values()
            assert np.array_equal(
                gdf_col_result_sorted.to_pandas().values,
                pd_col_result_sorted.values,
            )


def test_safe_merging_with_left_empty():
    rng = np.random.default_rng(seed=0)

    pairs = ("bcd", "b")
    pdf_left = pd.DataFrame()
    pdf_right = pd.DataFrame()
    for left_column in pairs[0]:
        pdf_left[left_column] = rng.integers(0, 10, 0)
    for right_column in pairs[1]:
        pdf_right[right_column] = rng.integers(0, 10, 5)
    gdf_left = cudf.from_pandas(pdf_left)
    gdf_right = cudf.from_pandas(pdf_right)

    pdf_result = pdf_left.merge(pdf_right)
    gdf_result = gdf_left.merge(gdf_right)
    # Simplify test because pandas does not consider empty Index and RangeIndex
    # to be equivalent. TODO: Allow empty Index objects to have equivalence.
    assert len(pdf_result) == len(gdf_result)


@pytest.mark.parametrize("how", ["left", "inner", "outer"])
@pytest.mark.parametrize("left_empty", [True, False])
@pytest.mark.parametrize("right_empty", [True, False])
def test_empty_joins(how, left_empty, right_empty):
    pdf = pd.DataFrame({"x": [1, 2, 3]})

    if left_empty:
        left = pdf.head(0)
    else:
        left = pdf
    if right_empty:
        right = pdf.head(0)
    else:
        right = pdf

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)

    expected = left.merge(right, how=how)
    result = gleft.merge(gright, how=how)
    assert len(expected) == len(result)


def test_merge_left_index_zero():
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[0, 1, 2, 3, 4, 5])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[0, 1, 2, 3, 4, 6]
    )
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, left_on="x", right_on="y")
    gd_merge = gleft.merge(gright, left_on="x", right_on="y")

    assert_join_results_equal(pd_merge, gd_merge, how="left")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
        {"left_index": True, "right_index": True},
    ],
)
def test_merge_left_right_index_left_right_on_zero_kwargs(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[0, 1, 2, 3, 4, 5])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[0, 1, 2, 3, 4, 6]
    )
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, **kwargs)
    gd_merge = gleft.merge(gright, **kwargs)
    assert_join_results_equal(pd_merge, gd_merge, how="left")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
        {"left_index": True, "right_index": True},
    ],
)
def test_merge_left_right_index_left_right_on_kwargs(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[1, 2, 3, 4, 5, 6])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[1, 2, 3, 4, 5, 7]
    )
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, **kwargs)
    gd_merge = gleft.merge(gright, **kwargs)
    assert_join_results_equal(pd_merge, gd_merge, how="left")


def test_indicator():
    gdf = cudf.DataFrame({"x": [1, 2, 1]})
    gdf.merge(gdf, indicator=False)

    with pytest.raises(NotImplementedError) as info:
        gdf.merge(gdf, indicator=True)

    assert "indicator=False" in str(info.value)


def test_merge_suffixes():
    pdf = cudf.DataFrame({"x": [1, 2, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 1]})
    assert_join_results_equal(
        gdf.merge(gdf, suffixes=("left", "right")),
        pdf.merge(pdf, suffixes=("left", "right")),
        how="left",
    )

    assert_exceptions_equal(
        lfunc=pdf.merge,
        rfunc=gdf.merge,
        lfunc_args_and_kwargs=([pdf], {"lsuffix": "left", "rsuffix": "right"}),
        rfunc_args_and_kwargs=([gdf], {"lsuffix": "left", "rsuffix": "right"}),
    )


def test_merge_left_on_right_on():
    left = pd.DataFrame({"xx": [1, 2, 3, 4, 5, 6]})
    right = pd.DataFrame({"xx": [10, 20, 30, 6, 5, 4]})

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)

    assert_join_results_equal(
        left.merge(right, on="xx"), gleft.merge(gright, on="xx"), how="left"
    )

    assert_join_results_equal(
        left.merge(right, left_on="xx", right_on="xx"),
        gleft.merge(gright, left_on="xx", right_on="xx"),
        how="left",
    )


def test_merge_on_index_retained():
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3, 4, 5]
    df["b"] = ["a", "b", "c", "d", "e"]
    df.index = [5, 3, 4, 2, 1]

    df2 = cudf.DataFrame()
    df2["a2"] = [1, 2, 3, 4, 5]
    df2["res"] = ["a", "b", "c", "d", "e"]

    pdf = df.to_pandas()
    pdf2 = df2.to_pandas()

    gdm = df.merge(df2, left_index=True, right_index=True, how="left")
    pdm = pdf.merge(pdf2, left_index=True, right_index=True, how="left")
    gdm["a2"] = gdm["a2"].astype("float64")
    assert_eq(gdm.sort_index(), pdm.sort_index())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
    ],
)
def test_merge_left_right_index_left_right_on_kwargs2(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
    right = pd.DataFrame({"y": [10, 20, 30]}, index=[1, 2, 30])
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)
    pd_merge = left.merge(right, **kwargs)
    if pd_merge.empty:
        assert gd_merge.empty


@pytest.mark.parametrize(
    "hows", [{"how": "inner"}, {"how": "left"}, {"how": "outer"}]
)
@pytest.mark.parametrize(
    "ons",
    [
        {"on": "a"},
        {"on": ["a", "b"]},
        {"on": ["b", "a"]},
        {"on": ["a", "aa", "b"]},
        {"on": ["b", "a", "aa"]},
    ],
)
def test_merge_sort(ons, hows):
    kwargs = {}
    kwargs.update(hows)
    kwargs.update(ons)
    kwargs["sort"] = True
    a = [4, 6, 9, 5, 2, 4, 1, 8, 1]
    b = [9, 8, 7, 8, 3, 9, 7, 9, 2]
    aa = [8, 9, 2, 9, 3, 1, 2, 3, 4]
    left = pd.DataFrame({"a": a, "b": b, "aa": aa})
    right = left.copy(deep=True)

    left.index = [6, 5, 4, 7, 5, 5, 5, 4, 4]
    right.index = [5, 4, 1, 9, 4, 3, 5, 4, 4]

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)

    pd_merge = left.merge(right, **kwargs)
    # require the join keys themselves to be sorted correctly
    # the non-key columns will NOT match pandas ordering
    assert_join_results_equal(
        pd_merge[kwargs["on"]], gd_merge[kwargs["on"]], how="left"
    )
    pd_merge = pd_merge.drop(kwargs["on"], axis=1)
    gd_merge = gd_merge.drop(kwargs["on"], axis=1)
    if not pd_merge.empty:
        # check to make sure the non join key columns are the same
        pd_merge = pd_merge.sort_values(list(pd_merge.columns)).reset_index(
            drop=True
        )
        gd_merge = gd_merge.sort_values(list(gd_merge.columns)).reset_index(
            drop=True
        )

    assert_join_results_equal(pd_merge, gd_merge, how="left")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_on": ["a"], "left_index": False, "right_index": True},
        {"right_on": ["b"], "left_index": True, "right_index": False},
    ],
)
def test_merge_sort_on_indexes(kwargs):
    left_index = kwargs["left_index"]
    right_index = kwargs["right_index"]
    kwargs["sort"] = True
    a = [4, 6, 9, 5, 2, 4, 1, 8, 1]
    left = pd.DataFrame({"a": a})
    right = pd.DataFrame({"b": a})

    left.index = [6, 5, 4, 7, 5, 5, 5, 4, 4]
    right.index = [5, 4, 1, 9, 4, 3, 5, 4, 4]

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)

    if left_index and right_index:
        check_if_sorted = gd_merge[["a", "b"]].to_pandas()
        check_if_sorted.index.name = "index"
        definitely_sorted = check_if_sorted.sort_values(["index", "a", "b"])
        definitely_sorted.index.name = None
        assert_eq(gd_merge, definitely_sorted)
    elif left_index:
        assert gd_merge["b"].is_monotonic_increasing
    elif right_index:
        assert gd_merge["a"].is_monotonic_increasing


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_join_datetimes_index(dtype):
    datetimes = pd.Series(pd.date_range("20010101", "20010102", freq="12h"))
    pdf_lhs = pd.DataFrame(index=[1, 0, 1, 2, 0, 0, 1])
    pdf_rhs = pd.DataFrame({"d": datetimes})
    gdf_lhs = cudf.from_pandas(pdf_lhs)
    gdf_rhs = cudf.from_pandas(pdf_rhs)

    gdf_rhs["d"] = gdf_rhs["d"].astype(dtype)

    pdf = pdf_lhs.join(pdf_rhs, sort=True)
    gdf = gdf_lhs.join(gdf_rhs, sort=True)

    assert gdf["d"].dtype == cudf.dtype(dtype)

    assert_join_results_equal(pdf, gdf, how="inner", check_dtype=False)


def test_join_with_different_names():
    left = pd.DataFrame({"a": [0, 1, 2.0, 3, 4, 5, 9]})
    right = pd.DataFrame({"b": [12, 5, 3, 9.0, 5], "c": [1, 2, 3, 4, 5.0]})
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, how="outer", left_on=["a"], right_on=["b"])
    gd_merge = gleft.merge(gright, how="outer", left_on=["a"], right_on=["b"])
    assert_join_results_equal(pd_merge, gd_merge, how="outer")


def test_join_same_name_different_order():
    left = pd.DataFrame({"a": [0, 0], "b": [1, 2]})
    right = pd.DataFrame({"a": [1, 2], "b": [0, 0]})
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, left_on=["a", "b"], right_on=["b", "a"])
    gd_merge = gleft.merge(gright, left_on=["a", "b"], right_on=["b", "a"])
    assert_join_results_equal(pd_merge, gd_merge, how="left")


def test_join_empty_table_dtype():
    left = pd.DataFrame({"a": []})
    right = pd.DataFrame({"b": [12, 5, 3, 9.0, 5], "c": [1, 2, 3, 4, 5.0]})
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, how="left", left_on=["a"], right_on=["b"])
    gd_merge = gleft.merge(gright, how="left", left_on=["a"], right_on=["b"])
    assert_eq(pd_merge["a"].dtype, gd_merge["a"].dtype)


@pytest.mark.parametrize("how", ["outer", "inner", "left", "right"])
@pytest.mark.parametrize(
    "column_a",
    [
        (
            pd.Series([None, 1, 2, 3, 4, 5, 6, 7], dtype=np.float64),
            pd.Series([8, 9, 10, 11, 12, None, 14, 15], dtype=np.float64),
        )
    ],
)
@pytest.mark.parametrize(
    "column_b",
    [
        (
            pd.Series([0, 1, 0, None, 1, 0, 0, 0], dtype=np.float64),
            pd.Series([None, 1, 2, 1, 2, 2, 0, 0], dtype=np.float64),
        )
    ],
)
@pytest.mark.parametrize(
    "column_c",
    [
        (
            pd.Series(["dog", "cat", "fish", "bug"] * 2),
            pd.Series(["bird", "cat", "mouse", "snake"] * 2),
        ),
        (
            pd.Series(["dog", "cat", "fish", "bug"] * 2).astype("category"),
            pd.Series(["bird", "cat", "mouse", "snake"] * 2).astype(
                "category"
            ),
        ),
    ],
)
def test_join_multi(how, column_a, column_b, column_c):
    index = ["b", "c"]
    df1 = pd.DataFrame()
    df1["a1"] = column_a[0]
    df1["b"] = column_b[0]
    df1["c"] = column_c[0]
    df1 = df1.set_index(index)
    gdf1 = cudf.from_pandas(df1)

    df2 = pd.DataFrame()
    df2["a2"] = column_a[1]
    df2["b"] = column_b[1]
    df2["c"] = column_c[1]
    df2 = df2.set_index(index)
    gdf2 = cudf.from_pandas(df2)

    gdf_result = gdf1.join(gdf2, how=how, sort=True)
    pdf_result = df1.join(df2, how=how, sort=True)

    # Make sure columns are in the same order
    columns = pdf_result.columns.values
    gdf_result = gdf_result[columns]
    pdf_result = pdf_result[columns]

    assert_join_results_equal(pdf_result, gdf_result, how="inner")


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "left_on": ["a", "b"],
            "right_on": ["a", "b"],
            "left_index": False,
            "right_index": False,
        },  # left and right on, no indices
        {
            "left_on": None,
            "right_on": None,
            "left_index": True,
            "right_index": True,
        },  # left_index and right_index, no on
        {
            "left_on": ["a", "b"],
            "right_on": None,
            "left_index": False,
            "right_index": True,
        },  # left on and right_index
        {
            "left_on": None,
            "right_on": ["a", "b"],
            "left_index": True,
            "right_index": False,
        },  # right_on and left_index
    ],
)
def test_merge_multi(kwargs):
    left = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 3, 5, 6],
            "b": [1, 3, 5, 7, 5, 9, 0],
            "c": ["o", "p", "q", "r", "s", "t", "u"],
            "d": ["v", "w", "x", "y", "z", "1", "2"],
        }
    )
    right = cudf.DataFrame(
        {
            "a": [0, 9, 3, 4, 3, 7, 8],
            "b": [2, 4, 5, 7, 5, 6, 8],
            "c": ["a", "b", "c", "d", "e", "f", "g"],
            "d": ["j", "i", "j", "k", "l", "m", "n"],
        }
    )

    if (
        kwargs["left_on"] is not None
        and kwargs["right_on"] is not None
        and kwargs["left_index"] is False
        and kwargs["right_index"] is False
    ):
        left = left.set_index(["c", "d"])
        right = right.set_index(["c", "d"])
    elif (
        kwargs["left_on"] is None
        and kwargs["right_on"] is None
        and kwargs["left_index"] is True
        and kwargs["right_index"] is True
    ):
        left = left.set_index(["a", "b"])
        right = right.set_index(["a", "b"])
    elif kwargs["left_on"] is not None and kwargs["right_index"] is True:
        left = left.set_index(["c", "d"])
        right = right.set_index(["a", "b"])
    elif kwargs["right_on"] is not None and kwargs["left_index"] is True:
        left = left.set_index(["a", "b"])
        right = right.set_index(["c", "d"])

    gleft = left.to_pandas()
    gright = right.to_pandas()

    kwargs["sort"] = True
    expect = gleft.merge(gright, **kwargs)
    got = left.merge(right, **kwargs)

    assert_eq(expect.sort_index().index, got.sort_index().index)

    expect.index = range(len(expect))
    got.index = range(len(got))
    expect = expect.sort_values(list(expect.columns))
    got = got.sort_values(list(got.columns))
    expect.index = range(len(expect))
    got.index = range(len(got))

    assert_join_results_equal(expect, got, how="left")


@pytest.mark.parametrize("dtype_l", INTEGER_TYPES)
@pytest.mark.parametrize("dtype_r", INTEGER_TYPES)
def test_typecast_on_join_int_to_int(dtype_l, dtype_r):
    other_data = ["a", "b", "c"]

    join_data_l = cudf.Series([1, 2, 3], dtype=dtype_l)
    join_data_r = cudf.Series([1, 2, 4], dtype=dtype_r)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = np.result_type(np.dtype(dtype_l), np.dtype(dtype_r))

    exp_join_data = [1, 2]
    exp_other_data = ["a", "b"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


@pytest.mark.parametrize("dtype_l", ["float32", "float64"])
@pytest.mark.parametrize("dtype_r", ["float32", "float64"])
def test_typecast_on_join_float_to_float(dtype_l, dtype_r):
    other_data = ["a", "b", "c", "d", "e", "f"]

    join_data_l = cudf.Series([1, 2, 3, 0.9, 4.5, 6], dtype=dtype_l)
    join_data_r = cudf.Series([1, 2, 3, 0.9, 4.5, 7], dtype=dtype_r)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = np.result_type(np.dtype(dtype_l), np.dtype(dtype_r))

    if dtype_l != dtype_r:
        exp_join_data = [1, 2, 3, 4.5]
        exp_other_data = ["a", "b", "c", "e"]
    else:
        exp_join_data = [1, 2, 3, 0.9, 4.5]
        exp_other_data = ["a", "b", "c", "d", "e"]

    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


@pytest.mark.parametrize("dtype_l", NUMERIC_TYPES)
@pytest.mark.parametrize("dtype_r", NUMERIC_TYPES)
def test_typecast_on_join_mixed_int_float(dtype_l, dtype_r):
    if (
        ("int" in dtype_l or "long" in dtype_l)
        and ("int" in dtype_r or "long" in dtype_r)
    ) or ("float" in dtype_l and "float" in dtype_r):
        pytest.skip("like types not tested in this function")

    other_data = ["a", "b", "c", "d", "e", "f"]

    join_data_l = cudf.Series([1, 2, 3, 0.9, 4.5, 6], dtype=dtype_l)
    join_data_r = cudf.Series([1, 2, 3, 0.9, 4.5, 7], dtype=dtype_r)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = np.result_type(np.dtype(dtype_l), np.dtype(dtype_r))

    exp_join_data = [1, 2, 3]
    exp_other_data = ["a", "b", "c"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


def test_typecast_on_join_no_float_round():
    other_data = ["a", "b", "c", "d", "e"]

    join_data_l = cudf.Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_r = cudf.Series([1, 2, 3, 4.01, 4.99], dtype="float32")

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = [1, 2, 3, 4, 5]
    exp_Bx = ["a", "b", "c", "d", "e"]
    exp_By = ["a", "b", "c", None, None]
    exp_join_col = cudf.Series(exp_join_data, dtype="float32")

    expect = cudf.DataFrame(
        {"join_col": exp_join_col, "B_x": exp_Bx, "B_y": exp_By}
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="left")

    assert_join_results_equal(expect, got, how="left")


@pytest.mark.parametrize(
    "dtypes",
    [
        (np.dtype("int8"), np.dtype("int16")),
        (np.dtype("int16"), np.dtype("int32")),
        (np.dtype("int32"), np.dtype("int64")),
        (np.dtype("uint8"), np.dtype("uint16")),
        (np.dtype("uint16"), np.dtype("uint32")),
        (np.dtype("uint32"), np.dtype("uint64")),
        (np.dtype("float32"), np.dtype("float64")),
        (np.dtype("int32"), np.dtype("float32")),
        (np.dtype("uint32"), np.dtype("float32")),
    ],
)
def test_typecast_on_join_overflow_unsafe(dtypes):
    dtype_l, dtype_r = dtypes
    if dtype_l.kind in {"i", "u"}:
        dtype_l_max = np.iinfo(dtype_l).max
    elif dtype_l.kind == "f":
        dtype_l_max = np.finfo(dtype_r).max

    lhs = cudf.DataFrame({"a": [1, 2, 3, 4, 5]}, dtype=dtype_l)
    rhs = cudf.DataFrame({"a": [1, 2, 3, 4, dtype_l_max + 1]}, dtype=dtype_r)

    p_lhs = lhs.to_pandas()
    p_rhs = rhs.to_pandas()

    with expect_warning_if(
        (dtype_l.kind == "f" and dtype_r.kind in {"i", "u"})
        or (dtype_l.kind in {"i", "u"} and dtype_r.kind == "f"),
        UserWarning,
    ):
        expect = p_lhs.merge(p_rhs, on="a", how="left")
    got = lhs.merge(rhs, on="a", how="left")

    # The dtypes here won't match exactly because pandas does some unsafe
    # conversions (with a warning that we are catching above) that we don't
    # want to match.
    assert_join_results_equal(expect, got, how="left", check_dtype=False)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(5, 2),
        Decimal64Dtype(7, 5),
        Decimal64Dtype(12, 7),
        Decimal128Dtype(20, 5),
    ],
)
def test_decimal_typecast_inner(dtype):
    other_data = ["a", "b", "c", "d", "e"]

    join_data_l = cudf.Series(["1.6", "9.5", "7.2", "8.7", "2.3"]).astype(
        dtype
    )
    join_data_r = cudf.Series(["1.6", "9.5", "7.2", "4.5", "2.3"]).astype(
        dtype
    )

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = ["1.6", "9.5", "7.2", "2.3"]
    exp_other_data = ["a", "b", "c", "e"]

    exp_join_col = cudf.Series(exp_join_data).astype(dtype)

    expected = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expected, got, how="inner")
    assert_eq(dtype, got["join_col"].dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(7, 3),
        Decimal64Dtype(9, 5),
        Decimal64Dtype(14, 10),
        Decimal128Dtype(21, 9),
    ],
)
def test_decimal_typecast_left(dtype):
    other_data = ["a", "b", "c", "d"]

    join_data_l = cudf.Series(["95.05", "384.26", "74.22", "1456.94"]).astype(
        dtype
    )
    join_data_r = cudf.Series(
        ["95.05", "62.4056", "74.22", "1456.9472"]
    ).astype(dtype)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = ["95.05", "74.22", "384.26", "1456.94"]
    exp_other_data_x = ["a", "c", "b", "d"]
    exp_other_data_y = ["a", "c", None, None]

    exp_join_col = cudf.Series(exp_join_data).astype(dtype)

    expected = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data_x,
            "B_y": exp_other_data_y,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="left")

    assert_join_results_equal(expected, got, how="left")
    assert_eq(dtype, got["join_col"].dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(7, 3),
        Decimal64Dtype(10, 5),
        Decimal64Dtype(18, 9),
        Decimal128Dtype(22, 8),
    ],
)
def test_decimal_typecast_outer(dtype):
    other_data = ["a", "b", "c"]
    join_data_l = cudf.Series(["741.248", "1029.528", "3627.292"]).astype(
        dtype
    )
    join_data_r = cudf.Series(["9284.103", "1029.528", "948.637"]).astype(
        dtype
    )
    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})
    exp_join_data = ["9284.103", "948.637", "1029.528", "741.248", "3627.292"]
    exp_other_data_x = [None, None, "b", "a", "c"]
    exp_other_data_y = ["a", "c", "b", None, None]
    exp_join_col = cudf.Series(exp_join_data).astype(dtype)
    expected = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data_x,
            "B_y": exp_other_data_y,
        }
    )
    got = gdf_l.merge(gdf_r, on="join_col", how="outer")

    assert_join_results_equal(expected, got, how="outer")
    assert_eq(dtype, got["join_col"].dtype)


@pytest.mark.parametrize(
    "dtype_l",
    [Decimal64Dtype(7, 3), Decimal64Dtype(9, 5)],
)
@pytest.mark.parametrize(
    "dtype_r",
    [Decimal64Dtype(8, 3), Decimal64Dtype(11, 6)],
)
def test_mixed_decimal_typecast(dtype_l, dtype_r):
    other_data = ["a", "b", "c", "d"]

    join_data_l = cudf.Series(["95.05", "34.6", "74.22", "14.94"]).astype(
        dtype_r
    )
    join_data_r = cudf.Series(["95.05", "62.4056", "74.22", "1.42"]).astype(
        dtype_l
    )

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    with pytest.raises(
        TypeError,
        match="Decimal columns can only be merged with decimal columns "
        "of the same precision and scale",
    ):
        gdf_l.merge(gdf_r, on="join_col", how="inner")


@pytest.mark.parametrize(
    "dtype_l",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "dtype_r",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_on_join_dt_to_dt(dtype_l, dtype_r):
    other_data = ["a", "b", "c", "d", "e"]
    join_data_l = cudf.Series(
        ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01", "2019-08-15"]
    ).astype(dtype_l)
    join_data_r = cudf.Series(
        ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01", "2019-08-16"]
    ).astype(dtype_r)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = max(np.dtype(dtype_l), np.dtype(dtype_r))

    exp_join_data = ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01"]
    exp_other_data = ["a", "b", "c", "d"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


@pytest.mark.parametrize("dtype_l", ["category", "str", "int32", "float32"])
@pytest.mark.parametrize("dtype_r", ["category", "str", "int32", "float32"])
def test_typecast_on_join_categorical(dtype_l, dtype_r):
    if not (dtype_l == "category" or dtype_r == "category"):
        pytest.skip("at least one side must be category for this set of tests")
    if dtype_l == "category" and dtype_r == "category":
        pytest.skip("Can't determine which categorical to use")

    other_data = ["a", "b", "c", "d", "e"]
    join_data_l = cudf.Series([1, 2, 3, 4, 5], dtype=dtype_l)
    join_data_r = cudf.Series([1, 2, 3, 4, 6], dtype=dtype_r)
    if dtype_l == "category":
        exp_dtype = join_data_l.dtype.categories.dtype
    elif dtype_r == "category":
        exp_dtype = join_data_r.dtype.categories.dtype

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = [1, 2, 3, 4]
    exp_other_data = ["a", "b", "c", "d"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")
    assert_join_results_equal(expect, got, how="inner")


def make_categorical_dataframe(categories, ordered=False):
    dtype = CategoricalDtype(categories=categories, ordered=ordered)
    data = cudf.Series(categories).astype(dtype)
    return cudf.DataFrame({"key": data})


def test_categorical_typecast_inner():
    # Inner join casting rules for categoricals

    # Equal categories, equal ordering -> common categorical
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)
    result = left.merge(right, how="inner", on="key")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=False)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(
        expect_data, result["key"], how="inner", check_categorical=False
    )

    # Equal categories, unequal ordering -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="inner", on="key")

    # Unequal categories
    # Neither ordered -> unordered categorical with intersection
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)

    result = left.merge(right, how="inner", on="key")

    expect_dtype = cudf.CategoricalDtype(categories=[2, 3], ordered=False)
    expect_data = cudf.Series([2, 3], dtype=expect_dtype, name="key")
    assert_join_results_equal(
        expect_data, result["key"], how="inner", check_categorical=False
    )

    # One is ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="inner", on="key")

    # Both are ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="inner", on="key")


def test_categorical_typecast_left():
    # TODO: generalize to right or write another test
    # Left join casting rules for categoricals

    # equal categories, neither ordered -> common dtype
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)

    result = left.merge(right, on="key", how="left")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=False)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="left")

    # equal categories, unequal ordering -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")
    with pytest.raises(TypeError):
        result = right.merge(left, on="key", how="left")

    # unequal categories neither ordered -> left dtype
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)

    result = left.merge(right, on="key", how="left")
    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=False)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="left")

    # unequal categories, unequal ordering -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")

    # unequal categories, right ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")

    # unequal categories, both ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")


def test_categorical_typecast_outer():
    # Outer join casting rules for categoricals

    # equal categories, neither ordered -> common dtype
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)
    result = left.merge(right, on="key", how="outer")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=False)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="outer")

    # equal categories, both ordered -> common dtype
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([1, 2, 3], ordered=True)
    result = left.merge(right, on="key", how="outer")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=True)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="outer")

    # equal categories, one ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="outer", on="key")
    with pytest.raises(TypeError):
        result = right.merge(left, how="outer", on="key")

    # unequal categories, neither ordered -> superset
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)
    result = left.merge(right, on="key", how="outer")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3, 4], ordered=False)
    expect_data = cudf.Series([1, 2, 3, 4], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="outer")

    # unequal categories, one ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="outer", on="key")
    with pytest.raises(TypeError):
        result = right.merge(left, how="outer", on="key")

    # unequal categories, both ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)
    with pytest.raises(TypeError):
        result = left.merge(right, how="outer", on="key")


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["str"])
def test_categorical_typecast_inner_one_cat(dtype):
    data = np.array([1, 2, 3], dtype=dtype)

    left = make_categorical_dataframe(data)
    right = left.astype(left["key"].dtype.categories.dtype)

    result = left.merge(right, on="key", how="inner")
    assert result["key"].dtype == left["key"].dtype.categories.dtype


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["str"])
def test_categorical_typecast_left_one_cat(dtype):
    data = np.array([1, 2, 3], dtype=dtype)

    left = make_categorical_dataframe(data)
    right = left.astype(left["key"].dtype.categories.dtype)

    result = left.merge(right, on="key", how="left")
    assert result["key"].dtype == left["key"].dtype


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["str"])
def test_categorical_typecast_outer_one_cat(dtype):
    data = np.array([1, 2, 3], dtype=dtype)

    left = make_categorical_dataframe(data)
    right = left.astype(left["key"].dtype.categories.dtype)

    result = left.merge(right, on="key", how="outer")
    assert result["key"].dtype == left["key"].dtype.categories.dtype


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (["a", "b"], ["a"]),
        (["a"], ["a", "b"]),
        (["a", "b"], ["b"]),
        (["b"], ["a", "b"]),
        (["a"], ["a"]),
    ],
)
@pytest.mark.parametrize("how", ["left", "right", "outer", "inner"])
@pytest.mark.parametrize("level", ["a", "b", 0, 1])
def test_index_join(lhs, rhs, how, level):
    l_pdf = pd.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_pdf = pd.DataFrame({"a": [1, 5, 4, 0], "b": [3, 9, 8, 4]})
    l_df = cudf.from_pandas(l_pdf)
    r_df = cudf.from_pandas(r_pdf)
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index

    expected = p_lhs.join(p_rhs, level=level, how=how).to_frame(index=False)
    got = g_lhs.join(g_rhs, level=level, how=how).to_frame(index=False)

    assert_join_results_equal(expected, got, how=how)


def test_index_join_corner_cases():
    l_pdf = pd.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_pdf = pd.DataFrame(
        {"a": [1, 5, 4, 0], "b": [3, 9, 8, 4], "c": [2, 3, 6, 0]}
    )
    l_df = cudf.from_pandas(l_pdf)
    r_df = cudf.from_pandas(r_pdf)

    # Join when column name doesn't match with level
    lhs = ["a", "b"]
    # level and rhs don't match
    rhs = ["c"]
    level = "b"
    how = "outer"
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = p_lhs.join(p_rhs, level=level, how=how).to_frame(index=False)
    got = g_lhs.join(g_rhs, level=level, how=how).to_frame(index=False)

    assert_join_results_equal(expected, got, how=how)

    # sort is supported only in case of two non-MultiIndex join
    # Join when column name doesn't match with level
    lhs = ["a"]
    # level and rhs don't match
    rhs = ["a"]
    level = "b"
    how = "left"
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = p_lhs.join(p_rhs, how=how, sort=True)
    got = g_lhs.join(g_rhs, how=how, sort=True)

    assert_join_results_equal(expected, got, how=how)

    # Pandas Index.join on categorical column returns generic column
    # but cudf will be returning a categorical column itself.
    lhs = ["a", "b"]
    rhs = ["a"]
    level = "a"
    how = "inner"
    l_df["a"] = l_df["a"].astype("category")
    r_df["a"] = r_df["a"].astype("category")
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = p_lhs.join(p_rhs, level=level, how=how).to_frame(index=False)
    got = g_lhs.join(g_rhs, level=level, how=how).to_frame(index=False)

    got["a"] = got["a"].astype(expected["a"].dtype)

    assert_join_results_equal(expected, got, how=how)


def test_index_join_exception_cases():
    l_df = cudf.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_df = cudf.DataFrame(
        {"a": [1, 5, 4, 0], "b": [3, 9, 8, 4], "c": [2, 3, 6, 0]}
    )

    # Join between two MultiIndex
    lhs = ["a", "b"]
    rhs = ["a", "c"]
    level = "a"
    how = "outer"
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index

    with pytest.raises(TypeError):
        g_lhs.join(g_rhs, level=level, how=how)

    # Improper level value, level should be an int or scalar value
    level = ["a"]
    rhs = ["a"]
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    with pytest.raises(ValueError):
        g_lhs.join(g_rhs, level=level, how=how)


def test_typecast_on_join_indexes():
    join_data_l = cudf.Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_r = cudf.Series([1, 2, 3, 4, 6], dtype="int32")
    other_data = ["a", "b", "c", "d", "e"]

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    gdf_l = gdf_l.set_index("join_col")
    gdf_r = gdf_r.set_index("join_col")

    exp_join_data = [1, 2, 3, 4]
    exp_other_data = ["a", "b", "c", "d"]

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_data,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index("join_col")

    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_join_results_equal(expect, got, how="inner")


def test_typecast_on_join_multiindices():
    join_data_l_0 = cudf.Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_l_1 = cudf.Series([2, 3, 4.1, 5.9, 6], dtype="float32")
    join_data_l_2 = cudf.Series([7, 8, 9, 0, 1], dtype="float32")

    join_data_r_0 = cudf.Series([1, 2, 3, 4, 5], dtype="int32")
    join_data_r_1 = cudf.Series([2, 3, 4, 5, 6], dtype="int32")
    join_data_r_2 = cudf.Series([7, 8, 9, 0, 0], dtype="float64")

    other_data = ["a", "b", "c", "d", "e"]

    gdf_l = cudf.DataFrame(
        {
            "join_col_0": join_data_l_0,
            "join_col_1": join_data_l_1,
            "join_col_2": join_data_l_2,
            "B": other_data,
        }
    )
    gdf_r = cudf.DataFrame(
        {
            "join_col_0": join_data_r_0,
            "join_col_1": join_data_r_1,
            "join_col_2": join_data_r_2,
            "B": other_data,
        }
    )

    gdf_l = gdf_l.set_index(["join_col_0", "join_col_1", "join_col_2"])
    gdf_r = gdf_r.set_index(["join_col_0", "join_col_1", "join_col_2"])

    exp_join_data_0 = cudf.Series([1, 2], dtype="int32")
    exp_join_data_1 = cudf.Series([2, 3], dtype="float64")
    exp_join_data_2 = cudf.Series([7, 8], dtype="float64")
    exp_other_data = cudf.Series(["a", "b"])

    expect = cudf.DataFrame(
        {
            "join_col_0": exp_join_data_0,
            "join_col_1": exp_join_data_1,
            "join_col_2": exp_join_data_2,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index(["join_col_0", "join_col_1", "join_col_2"])
    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_join_results_equal(expect, got, how="inner")


def test_typecast_on_join_indexes_matching_categorical():
    join_data_l = cudf.Series(["a", "b", "c", "d", "e"], dtype="category")
    join_data_r = cudf.Series(["a", "b", "c", "d", "e"], dtype="str")
    other_data = [1, 2, 3, 4, 5]

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    gdf_l = gdf_l.set_index("join_col")
    gdf_r = gdf_r.set_index("join_col")

    exp_join_data = ["a", "b", "c", "d", "e"]
    exp_other_data = [1, 2, 3, 4, 5]

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_data,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index("join_col")
    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_join_results_equal(expect, got, how="inner")


@pytest.mark.parametrize(
    "lhs",
    [
        cudf.Series([1, 2, 3], name="a"),
        cudf.DataFrame({"a": [2, 3, 4], "c": [4, 5, 6]}),
    ],
)
@pytest.mark.parametrize(
    "rhs",
    [
        cudf.Series([1, 2, 3], name="b"),
        cudf.DataFrame({"b": [2, 3, 4], "c": [4, 5, 6]}),
    ],
)
@pytest.mark.parametrize(
    "how", ["left", "inner", "outer", "leftanti", "leftsemi"]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_on": "a", "right_on": "b"},
        {"left_index": True, "right_on": "b"},
        {"left_on": "a", "right_index": True},
        {"left_index": True, "right_index": True},
    ],
)
def test_series_dataframe_mixed_merging(lhs, rhs, how, kwargs):
    if how in ("leftsemi", "leftanti") and (
        kwargs.get("left_index") or kwargs.get("right_index")
    ):
        pytest.skip("Index joins not compatible with leftsemi and leftanti")

    check_lhs = lhs.copy()
    check_rhs = rhs.copy()
    if isinstance(lhs, cudf.Series):
        check_lhs = lhs.to_frame()
    if isinstance(rhs, cudf.Series):
        check_rhs = rhs.to_frame()

    expect = cudf.merge(check_lhs, check_rhs, how=how, **kwargs)
    got = cudf.merge(lhs, rhs, how=how, **kwargs)

    assert_join_results_equal(expect, got, how=how)


@pytest.mark.xfail(reason="Cannot sort values of list dtype")
@pytest.mark.parametrize(
    "how", ["left", "inner", "right", "leftanti", "leftsemi"]
)
def test_merge_with_lists(how):
    pd_left = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [[1, 2, 3], [4, 5], None, [6], [7, 8, None], []],
            "c": ["a", "b", "c", "d", "e", "f"],
        }
    )
    pd_right = pd.DataFrame(
        {
            "a": [4, 3, 2, 1, 0, -1],
            "d": [[[1, 2], None], [], [[3, 4]], None, [[5], [6, 7]], [[8]]],
        }
    )

    gd_left = cudf.from_pandas(pd_left)
    gd_right = cudf.from_pandas(pd_right)

    expect = pd_left.merge(pd_right, on="a")
    got = gd_left.merge(gd_right, on="a")

    assert_join_results_equal(expect, got, how=how)


def test_join_renamed_index():
    df = cudf.DataFrame(
        {0: [1, 2, 3, 4, 5], 1: [1, 2, 3, 4, 5], "c": [1, 2, 3, 4, 5]}
    ).set_index([0, 1])
    df.index.names = ["a", "b"]  # doesn't actually change df._index._data

    expect = df.to_pandas().merge(
        df.to_pandas(), left_index=True, right_index=True
    )
    got = df.merge(df, left_index=True, right_index=True, how="inner")
    assert_join_results_equal(expect, got, how="inner")


@pytest.mark.parametrize(
    "lhs_col, lhs_idx, rhs_col, rhs_idx, on",
    [
        (["A", "B"], "L0", ["B", "C"], "L0", ["B"]),
        (["A", "B"], "L0", ["B", "C"], "L0", ["L0"]),
        (["A", "B"], "L0", ["B", "C"], "L0", ["B", "L0"]),
        (["A", "B"], "L0", ["C", "L0"], "A", ["A"]),
        (["A", "B"], "L0", ["C", "L0"], "A", ["L0"]),
        (["A", "B"], "L0", ["C", "L0"], "A", ["A", "L0"]),
    ],
)
@pytest.mark.parametrize(
    "how", ["left", "inner", "right", "outer", "leftanti", "leftsemi"]
)
def test_join_merge_with_on(lhs_col, lhs_idx, rhs_col, rhs_idx, on, how):
    lhs_data = {col_name: [4, 5, 6] for col_name in lhs_col}
    lhs_index = cudf.Index([0, 1, 2], name=lhs_idx)

    rhs_data = {col_name: [4, 5, 6] for col_name in rhs_col}
    rhs_index = cudf.Index([2, 3, 4], name=rhs_idx)

    gd_left = cudf.DataFrame(lhs_data, lhs_index)
    gd_right = cudf.DataFrame(rhs_data, rhs_index)
    pd_left = gd_left.to_pandas()
    pd_right = gd_right.to_pandas()

    expect = pd_left.merge(pd_right, on=on).sort_index(axis=1, ascending=False)
    got = gd_left.merge(gd_right, on=on).sort_index(axis=1, ascending=False)

    assert_join_results_equal(expect, got, how=how)


@pytest.mark.parametrize(
    "on",
    ["A", "L0"],
)
@pytest.mark.parametrize(
    "how", ["left", "inner", "right", "outer", "leftanti", "leftsemi"]
)
def test_join_merge_invalid_keys(on, how):
    gd_left = cudf.DataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6]}, index=cudf.Index([0, 1, 2], name="C")
    )
    gd_right = cudf.DataFrame(
        {"D": [2, 3, 4], "E": [7, 8, 0]}, index=cudf.Index([0, 2, 4], name="F")
    )
    pd_left = gd_left.to_pandas()
    pd_right = gd_right.to_pandas()

    with pytest.raises(KeyError):
        pd_left.merge(pd_right, on=on)
        gd_left.merge(gd_right, on=on)


@pytest.mark.parametrize(
    "str_data",
    [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]],
)
@pytest.mark.parametrize("num_keys", [1, 2, 3])
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_string_join_key(str_data, num_keys, how):
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()
    for i in range(num_keys):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = cudf.Series(str_data, dtype="str")
    pdf["a"] = other_data
    gdf["a"] = other_data
    if len(other_data) == 0:
        pdf["a"] = pdf["a"].astype("str")
    pdf2 = pdf.copy()
    gdf2 = gdf.copy()

    expect = pdf.merge(pdf2, on=list(range(num_keys)), how=how)
    got = gdf.merge(gdf2, on=list(range(num_keys)), how=how)

    if len(expect) == 0 and len(got) == 0:
        expect = expect.reset_index(drop=True)
        got = got[expect.columns]  # reorder columns

    if how == "right":
        got = got[expect.columns]  # reorder columns

    assert_join_results_equal(expect, got, how=how)


@pytest.mark.parametrize(
    "str_data_nulls",
    [
        ["a", "b", "c"],
        ["a", "b", "f", "g"],
        ["f", "g", "h", "i", "j"],
        ["f", "g", "h"],
        [None, None, None, None, None],
        [],
    ],
)
def test_string_join_key_nulls(str_data_nulls):
    str_data = ["a", "b", "c", "d", "e"]
    other_data = [1, 2, 3, 4, 5]

    other_data_nulls = [6, 7, 8, 9, 10][: len(str_data_nulls)]

    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()
    pdf["key"] = pd.Series(str_data, dtype="str")
    gdf["key"] = cudf.Series(str_data, dtype="str")
    pdf["vals"] = other_data
    gdf["vals"] = other_data

    pdf2 = pd.DataFrame()
    gdf2 = cudf.DataFrame()
    pdf2["key"] = pd.Series(str_data_nulls, dtype="str")
    gdf2["key"] = cudf.Series(str_data_nulls, dtype="str")
    pdf2["vals"] = pd.Series(other_data_nulls, dtype="int64")
    gdf2["vals"] = cudf.Series(other_data_nulls, dtype="int64")

    expect = pdf.merge(pdf2, on="key", how="left")
    got = gdf.merge(gdf2, on="key", how="left")
    got["vals_y"] = got["vals_y"].fillna(-1)

    if len(expect) == 0 and len(got) == 0:
        expect = expect.reset_index(drop=True)
        got = got[expect.columns]

    expect["vals_y"] = expect["vals_y"].fillna(-1).astype("int64")

    assert_join_results_equal(expect, got, how="left")


@pytest.mark.parametrize(
    "str_data", [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]]
)
@pytest.mark.parametrize("num_cols", [1, 2, 3])
@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_string_join_non_key(str_data, num_cols, how):
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()
    for i in range(num_cols):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = cudf.Series(str_data, dtype="str")
    pdf["a"] = other_data
    gdf["a"] = other_data
    if len(other_data) == 0:
        pdf["a"] = pdf["a"].astype("str")

    pdf2 = pdf.copy()
    gdf2 = gdf.copy()

    expect = pdf.merge(pdf2, on=["a"], how=how)
    got = gdf.merge(gdf2, on=["a"], how=how)

    if len(expect) == 0 and len(got) == 0:
        expect = expect.reset_index(drop=True)
        got = got[expect.columns]

    if how == "right":
        got = got[expect.columns]  # reorder columns

    assert_join_results_equal(expect, got, how=how)


@pytest.mark.parametrize(
    "str_data_nulls",
    [
        ["a", "b", "c"],
        ["a", "b", "f", "g"],
        ["f", "g", "h", "i", "j"],
        ["f", "g", "h"],
        [None, None, None, None, None],
        [],
    ],
)
def test_string_join_non_key_nulls(str_data_nulls):
    str_data = ["a", "b", "c", "d", "e"]
    other_data = [1, 2, 3, 4, 5]

    other_data_nulls = [6, 7, 8, 9, 10][: len(str_data_nulls)]

    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()
    pdf["vals"] = pd.Series(str_data, dtype="str")
    gdf["vals"] = cudf.Series(str_data, dtype="str")
    pdf["key"] = other_data
    gdf["key"] = other_data

    pdf2 = pd.DataFrame()
    gdf2 = cudf.DataFrame()
    pdf2["vals"] = pd.Series(str_data_nulls, dtype="str")
    gdf2["vals"] = cudf.Series(str_data_nulls, dtype="str")
    pdf2["key"] = pd.Series(other_data_nulls, dtype="int64")
    gdf2["key"] = cudf.Series(other_data_nulls, dtype="int64")

    expect = pdf.merge(pdf2, on="key", how="left")
    got = gdf.merge(gdf2, on="key", how="left")

    if len(expect) == 0 and len(got) == 0:
        expect = expect.reset_index(drop=True)
        got = got[expect.columns]

    assert_join_results_equal(expect, got, how="left")


def test_string_join_values_nulls():
    left_dict = [
        {"b": "MATCH 1", "a": 1.0},
        {"b": "MATCH 1", "a": 1.0},
        {"b": "LEFT NO MATCH 1", "a": -1.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "MATCH 1", "a": 1.0},
        {"b": "MATCH 1", "a": 1.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "LEFT NO MATCH 2", "a": -2.0},
        {"b": "MATCH 3", "a": 3.0},
        {"b": "MATCH 3", "a": 3.0},
    ]

    right_dict = [
        {"b": "RIGHT NO MATCH 1", "c": -1.0},
        {"b": "MATCH 3", "c": 3.0},
        {"b": "MATCH 2", "c": 2.0},
        {"b": "RIGHT NO MATCH 2", "c": -2.0},
        {"b": "RIGHT NO MATCH 3", "c": -3.0},
        {"b": "MATCH 1", "c": 1.0},
    ]

    left_pdf = pd.DataFrame(left_dict)
    right_pdf = pd.DataFrame(right_dict)

    left_gdf = cudf.DataFrame.from_pandas(left_pdf)
    right_gdf = cudf.DataFrame.from_pandas(right_pdf)

    expect = left_pdf.merge(right_pdf, how="left", on="b")
    got = left_gdf.merge(right_gdf, how="left", on="b")

    expect = expect.sort_values(by=["a", "b", "c"]).reset_index(drop=True)
    got = got.sort_values(by=["a", "b", "c"]).reset_index(drop=True)

    assert_join_results_equal(expect, got, how="left")


@pytest.mark.parametrize(
    "left_on,right_on",
    [
        *product(["a", "b", "c"], ["a", "b"]),
        *zip(combinations(["a", "b", "c"], 2), repeat(["a", "b"])),
    ],
)
def test_merge_mixed_index_columns(left_on, right_on):
    left = pd.DataFrame({"a": [1, 2, 1, 2], "b": [2, 3, 3, 4]}).set_index("a")
    right = pd.DataFrame({"a": [1, 2, 1, 3], "b": [2, 30, 3, 4]}).set_index(
        "a"
    )

    left["c"] = 10

    expect = left.merge(right, left_on=left_on, right_on=right_on, how="outer")
    cleft = cudf.from_pandas(left)
    cright = cudf.from_pandas(right)
    got = cleft.merge(cright, left_on=left_on, right_on=right_on, how="outer")
    assert_join_results_equal(expect, got, how="outer")


def test_merge_multiindex_columns():
    lhs = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    lhs.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
    rhs = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    rhs.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "z")])
    expect = lhs.merge(rhs, on=[("a", "x")], how="inner")

    lhs = cudf.from_pandas(lhs)
    rhs = cudf.from_pandas(rhs)
    got = lhs.merge(rhs, on=[("a", "x")], how="inner")

    assert_join_results_equal(expect, got, how="inner")


def test_join_multiindex_empty():
    lhs = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}, index=["a", "b", "c"])
    lhs.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
    rhs = pd.DataFrame(index=["a", "c", "d"])
    g_lhs = cudf.from_pandas(lhs)
    g_rhs = cudf.from_pandas(rhs)
    assert_exceptions_equal(
        lfunc=lhs.join,
        rfunc=g_lhs.join,
        lfunc_args_and_kwargs=([rhs], {"how": "inner"}),
        rfunc_args_and_kwargs=([g_rhs], {"how": "inner"}),
        check_exception_type=False,
    )


def test_join_on_index_with_duplicate_names():
    # although index levels with duplicate names are poorly supported
    # overall, we *should* be able to join on them:
    lhs = pd.DataFrame({"a": [1, 2, 3]})
    rhs = pd.DataFrame({"b": [1, 2, 3]})
    lhs.index = pd.MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)], names=["x", "x"]
    )
    rhs.index = pd.MultiIndex.from_tuples(
        [(1, 1), (1, 3), (2, 1)], names=["x", "x"]
    )
    expect = lhs.join(rhs, how="inner")

    lhs = cudf.from_pandas(lhs)
    rhs = cudf.from_pandas(rhs)
    got = lhs.join(rhs, how="inner")

    assert_join_results_equal(expect, got, how="inner")


def test_join_redundant_params():
    lhs = cudf.DataFrame(
        {"a": [1, 2, 3], "c": [2, 3, 4]}, index=cudf.Index([0, 1, 2], name="c")
    )
    rhs = cudf.DataFrame(
        {"b": [1, 2, 3]}, index=cudf.Index([0, 1, 2], name="a")
    )
    with pytest.raises(ValueError):
        lhs.merge(rhs, on="a", left_index=True)
    with pytest.raises(ValueError):
        lhs.merge(rhs, left_on="a", left_index=True, right_index=True)
    with pytest.raises(ValueError):
        lhs.merge(rhs, right_on="a", left_index=True, right_index=True)
    with pytest.raises(ValueError):
        lhs.merge(rhs, left_on="c", right_on="b")


def test_join_multiindex_index():
    # test joining a MultiIndex with an Index with overlapping name
    lhs = (
        cudf.DataFrame({"a": [2, 3, 1], "b": [3, 4, 2]})
        .set_index(["a", "b"])
        .index
    )
    rhs = cudf.DataFrame({"a": [1, 4, 3]}).set_index("a").index
    expect = lhs.to_pandas().join(rhs.to_pandas(), how="inner")
    got = lhs.join(rhs, how="inner")
    assert_join_results_equal(expect, got, how="inner")


def test_dataframe_join_on():
    """Verify that specifying the on parameter gives a NotImplementedError."""
    df = cudf.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError):
        df.join(df, on="a")


def test_index_join_return_indexers_notimplemented():
    index = cudf.RangeIndex(start=0, stop=20, step=2)
    other = cudf.Index([4, 4, 3, 3])
    with pytest.raises(NotImplementedError):
        index.join(other, how="left", return_indexers=True)


@pytest.mark.parametrize("how", ["inner", "outer"])
def test_index_join_names(request, how):
    idx1 = cudf.Index([10, 1, 2, 4, 2, 1], name="a")
    idx2 = cudf.Index([-10, 2, 3, 1, 2], name="b")
    request.applymarker(
        pytest.mark.xfail(
            reason="https://github.com/pandas-dev/pandas/issues/57065",
        )
    )
    pidx1 = idx1.to_pandas()
    pidx2 = idx2.to_pandas()

    expected = pidx1.join(pidx2, how=how)
    actual = idx1.join(idx2, how=how)
    assert_join_results_equal(actual, expected, how=how)


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_join_datetime_timedelta_error(dtype):
    df1 = cudf.DataFrame({"a": cudf.Series([10, 20, 30], dtype=dtype)})
    df2 = df1.astype("int")

    with pytest.raises(TypeError):
        df1.merge(df2)


@pytest.mark.parametrize("dtype1", TIMEDELTA_TYPES)
@pytest.mark.parametrize("dtype2", TIMEDELTA_TYPES)
def test_merge_timedelta_types(dtype1, dtype2):
    df1 = cudf.DataFrame({"a": cudf.Series([10, 20, 30], dtype=dtype1)})
    df2 = cudf.DataFrame({"a": cudf.Series([20, 500, 33240], dtype=dtype2)})

    pdf1 = df1.to_pandas()
    pdf2 = df2.to_pandas()
    actual = df1.merge(df2)
    expected = pdf1.merge(pdf2)

    # Pandas is materializing the index, which is unnecessary
    # hence the special handling.
    assert_eq(
        actual,
        expected,
        check_index_type=False
        if isinstance(actual.index, cudf.RangeIndex)
        and isinstance(expected.index, pd.Index)
        else True,
        check_dtype=len(actual) > 0,
    )
