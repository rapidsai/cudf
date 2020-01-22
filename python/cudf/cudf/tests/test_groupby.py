# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import cudf
from cudf.core import DataFrame, Series
from cudf.tests.utils import assert_eq

_now = np.datetime64("now")
_tomorrow = _now + np.timedelta64(1, "D")
_now = np.int64(_now.astype("datetime64[ns]"))
_tomorrow = np.int64(_tomorrow.astype("datetime64[ns]"))


def make_frame(
    dataframe_class,
    nelem,
    seed=0,
    extra_levels=(),
    extra_vals=(),
    with_datetime=False,
):
    np.random.seed(seed)

    df = dataframe_class()

    df["x"] = np.random.randint(0, 5, nelem)
    df["y"] = np.random.randint(0, 3, nelem)
    for lvl in extra_levels:
        df[lvl] = np.random.randint(0, 2, nelem)

    df["val"] = np.random.random(nelem)
    for val in extra_vals:
        df[val] = np.random.random(nelem)

    if with_datetime:
        df["datetime"] = np.random.randint(
            _now, _tomorrow, nelem, dtype=np.int64
        ).astype("datetime64[ns]")

    return df


def get_methods():
    for method in ["cudf", "hash"]:
        yield method


def get_nelem():
    for elem in [2, 3, 1000]:
        yield elem


@pytest.fixture
def gdf():
    return DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})


@pytest.fixture
def pdf(gdf):
    return gdf.to_pandas()


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_as_index_single_agg(pdf, gdf, as_index):
    gdf = gdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    pdf = pdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_as_index_multiindex(pdf, gdf, as_index):
    pdf = pd.DataFrame(
        {"a": [1, 2, 1], "b": [3, 3, 3], "c": [2, 2, 3], "d": [3, 1, 2]}
    )
    gdf = cudf.from_pandas(pdf)

    gdf = gdf.groupby(["a", "b"], as_index=as_index).agg({"c": "mean"})
    pdf = pdf.groupby(["a", "b"], as_index=as_index).agg({"c": "mean"})

    if as_index:
        assert_eq(pdf, gdf)
    else:
        # column names don't match - check just the values
        for gcol, pcol in zip(gdf, pdf):
            assert_array_equal(gdf[gcol].to_array(), pdf[pcol].values)


def test_groupby_default(pdf, gdf):
    gdf = gdf.groupby("y").agg({"x": "mean"})
    pdf = pdf.groupby("y").agg({"x": "mean"})
    assert_eq(pdf, gdf)


def test_group_keys_true(pdf, gdf):
    gdf = gdf.groupby("y", group_keys=True).sum()
    pdf = pdf.groupby("y", group_keys=True).sum()
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_getitem_getattr(as_index):
    pdf = pd.DataFrame({"x": [1, 3, 1], "y": [1, 2, 3], "z": [1, 4, 5]})
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.groupby("x")["y"].sum(), gdf.groupby("x")["y"].sum())
    assert_eq(pdf.groupby("x").y.sum(), gdf.groupby("x").y.sum())
    assert_eq(pdf.groupby("x")[["y"]].sum(), gdf.groupby("x")[["y"]].sum())
    assert_eq(
        pdf.groupby(["x", "y"], as_index=as_index).sum(),
        gdf.groupby(["x", "y"], as_index=as_index).sum(),
    )


@pytest.mark.parametrize("nelem", get_nelem())
@pytest.mark.parametrize("method", get_methods())
def test_groupby_mean(nelem, method):
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], method=method)
        .mean()
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem).groupby(["x", "y"]).mean()
    )

    if method == "cudf":
        got = np.sort(got_df["val"].to_array())
        expect = np.sort(expect_df["val"].values)
        np.testing.assert_array_almost_equal(expect, got)
    else:
        assert_eq(got_df, expect_df)


@pytest.mark.parametrize("nelem", get_nelem())
@pytest.mark.parametrize("method", get_methods())
def test_groupby_mean_3level(nelem, method):
    lvls = "z"
    bys = list("xyz")
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_levels=lvls)
        .groupby(bys, method=method)
        .mean()
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_levels=lvls)
        .groupby(bys)
        .mean()
    )

    if method == "cudf":
        got = np.sort(got_df["val"].to_array())
        expect = np.sort(expect_df["val"].values)
        np.testing.assert_array_almost_equal(expect, got)
    else:
        assert_eq(got_df, expect_df)


@pytest.mark.parametrize("nelem", get_nelem())
@pytest.mark.parametrize("method", get_methods())
def test_groupby_agg_mean_min(nelem, method):
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], method=method)
        .agg(["mean", "min"])
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem)
        .groupby(["x", "y"])
        .agg(["mean", "min"])
    )

    if method == "cudf":
        got_mean = np.sort(got_df["val_mean"].to_array())
        got_min = np.sort(got_df["val_min"].to_array())
        expect_mean = np.sort(expect_df["val", "mean"].values)
        expect_min = np.sort(expect_df["val", "min"].values)
        # verify
        np.testing.assert_array_almost_equal(expect_mean, got_mean)
        np.testing.assert_array_almost_equal(expect_min, got_min)
    else:
        assert_eq(expect_df, got_df)


@pytest.mark.parametrize("nelem", get_nelem())
@pytest.mark.parametrize("method", get_methods())
def test_groupby_agg_min_max_dictargs(nelem, method):
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"], method=method)
        .agg({"a": "min", "b": "max"})
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"])
        .agg({"a": "min", "b": "max"})
    )

    if method == "cudf":
        got_min = np.sort(got_df["a"].to_array())
        got_max = np.sort(got_df["b"].to_array())
        expect_min = np.sort(expect_df["a"].values)
        expect_max = np.sort(expect_df["b"].values)
        # verify
        np.testing.assert_array_almost_equal(expect_min, got_min)
        np.testing.assert_array_almost_equal(expect_max, got_max)
    else:
        assert_eq(expect_df, got_df)


@pytest.mark.parametrize("method", get_methods())
def test_groupby_cats(method):
    df = DataFrame()
    df["cats"] = pd.Categorical(list("aabaacaab"))
    df["vals"] = np.random.random(len(df))

    cats = np.asarray(list(df["cats"]))
    vals = df["vals"].to_array()

    grouped = df.groupby(["cats"], method=method, as_index=False).mean()

    got_vals = grouped["vals"]

    got_cats = grouped["cats"]

    for c, v in zip(got_cats, got_vals):
        print(c, v)
        expect = vals[cats == c].mean()
        np.testing.assert_almost_equal(v, expect)


def test_groupby_iterate_groups():
    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df["key1"] = np.random.randint(0, 3, nelem)
    df["key2"] = np.random.randint(0, 2, nelem)
    df["val1"] = np.random.random(nelem)
    df["val2"] = np.random.random(nelem)

    def assert_values_equal(arr):
        np.testing.assert_array_equal(arr[0], arr)

    for grp in df.groupby(["key1", "key2"], method="cudf"):
        pddf = grp.to_pandas()
        for k in "key1,key2".split(","):
            assert_values_equal(pddf[k].values)


def test_groupby_as_df():
    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df["key1"] = np.random.randint(0, 3, nelem)
    df["key2"] = np.random.randint(0, 2, nelem)
    df["val1"] = np.random.random(nelem)
    df["val2"] = np.random.random(nelem)

    def assert_values_equal(arr):
        np.testing.assert_array_equal(arr[0], arr)

    df, segs = df.groupby(["key1", "key2"], method="cudf").as_df()
    for s, e in zip(segs, list(segs[1:]) + [None]):
        grp = df[s:e]
        pddf = grp.to_pandas()
        for k in "key1,key2".split(","):
            assert_values_equal(pddf[k].values)


def test_groupby_apply():
    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df["key1"] = np.random.randint(0, 3, nelem)
    df["key2"] = np.random.randint(0, 2, nelem)
    df["val1"] = np.random.random(nelem)
    df["val2"] = np.random.random(nelem)

    expect_grpby = df.to_pandas().groupby(["key1", "key2"], as_index=False)
    got_grpby = df.groupby(["key1", "key2"], method="cudf")

    def foo(df):
        df["out"] = df["val1"] + df["val2"]
        return df

    expect = expect_grpby.apply(foo)
    expect = expect.sort_values(["key1", "key2"]).reset_index(drop=True)

    got = got_grpby.apply(foo).to_pandas()
    pd.util.testing.assert_frame_equal(expect, got)


def test_groupby_apply_grouped():
    from numba import cuda

    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df["key1"] = np.random.randint(0, 3, nelem)
    df["key2"] = np.random.randint(0, 2, nelem)
    df["val1"] = np.random.random(nelem)
    df["val2"] = np.random.random(nelem)

    expect_grpby = df.to_pandas().groupby(["key1", "key2"], as_index=False)
    got_grpby = df.groupby(["key1", "key2"], method="cudf")

    def foo(key1, val1, com1, com2):
        for i in range(cuda.threadIdx.x, len(key1), cuda.blockDim.x):
            com1[i] = key1[i] * 10000 + val1[i]
            com2[i] = i

    got = got_grpby.apply_grouped(
        foo,
        incols=["key1", "val1"],
        outcols={"com1": np.float64, "com2": np.int32},
        tpb=8,
    )

    got = got.to_pandas()

    # Get expected result by emulating the operation in pandas
    def emulate(df):
        df["com1"] = df.key1 * 10000 + df.val1
        df["com2"] = np.arange(len(df), dtype=np.int32)
        return df

    expect = expect_grpby.apply(emulate)
    expect = expect.sort_values(["key1", "key2"]).reset_index(drop=True)

    pd.util.testing.assert_frame_equal(expect, got)


@pytest.mark.parametrize("nelem", [100, 500])
@pytest.mark.parametrize(
    "func", ["mean", "std", "var", "min", "max", "count", "sum"]
)
@pytest.mark.parametrize("method", get_methods())
def test_groupby_cudf_2keys_agg(nelem, func, method):
    # skip unimplemented aggs:
    if func in ["var", "std"]:
        if method in ["hash", "sort"]:
            pytest.skip()

    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], method=method)
        .agg(func)
    )

    got_agg = np.sort(got_df["val"].to_array())
    # pandas
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem).groupby(["x", "y"]).agg(func)
    )
    if method == "cudf":
        expect_agg = np.sort(expect_df["val"].values)
        # verify
        np.testing.assert_array_almost_equal(expect_agg, got_agg)
    else:
        check_dtype = False if func == "count" else True
        assert_eq(got_df, expect_df, check_dtype=check_dtype)


@pytest.mark.parametrize("agg", ["min", "max", "count", "sum", "mean"])
def test_series_groupby(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2)
    gg = g.groupby(g // 2)
    sa = getattr(sg, agg)()
    ga = getattr(gg, agg)()
    check_dtype = False if agg == "count" else True
    assert_eq(sa, ga, check_dtype=check_dtype)


@pytest.mark.parametrize("agg", ["min", "max", "count", "sum", "mean"])
def test_series_groupby_agg(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2).agg(agg)
    gg = g.groupby(g // 2).agg(agg)
    check_dtype = False if agg == "count" else True
    assert_eq(sg, gg, check_dtype=check_dtype)


@pytest.mark.parametrize("agg", ["min", "max", "count", "sum", "mean"])
def test_groupby_level_zero(agg):
    pdf = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 1])
    gdf = DataFrame.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    check_dtype = False if agg == "count" else True
    assert_eq(pdresult, gdresult, check_dtype=check_dtype)


@pytest.mark.parametrize("agg", ["min", "max", "count", "sum", "mean"])
def test_groupby_series_level_zero(agg):
    pdf = pd.Series([1, 2, 3], index=[0, 1, 1])
    gdf = Series.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    check_dtype = False if agg == "count" else True
    assert_eq(pdresult, gdresult, check_dtype=check_dtype)


def test_groupby_column_name():
    pdf = pd.DataFrame({"xx": [1.0, 2.0, 3.0], "yy": [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    g = gdf.groupby("yy")
    p = pdf.groupby("yy")
    gxx = g["xx"].sum()
    pxx = p["xx"].sum()
    assert_eq(pxx, gxx)


def test_groupby_column_numeral():
    pdf = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    p = pdf.groupby(1)
    g = gdf.groupby(1)
    pxx = p[0].sum()
    gxx = g[0].sum()
    assert_eq(pxx, gxx)

    pdf = pd.DataFrame({0.5: [1.0, 2.0, 3.0], 1.5: [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    p = pdf.groupby(1.5)
    g = gdf.groupby(1.5)
    pxx = p[0.5].sum()
    gxx = g[0.5].sum()
    assert_eq(pxx, gxx)


@pytest.mark.parametrize(
    "series",
    [[0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 2, 3], [4, 3, 2], [0, 2, 0]],
)  # noqa: E501
def test_groupby_external_series(series):
    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]})
    gdf = DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_eq(pxx, gxx)


@pytest.mark.parametrize("series", [[0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
def test_groupby_external_series_incorrect_length(series):
    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]})
    gdf = DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_eq(pxx, gxx)


@pytest.mark.parametrize(
    "level", [0, 1, "a", "b", [0, 1], ["a", "b"], ["a", 1], -1, [-1, -2]]
)
def test_groupby_levels(level):
    idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 2)], names=("a", "b"))
    pdf = pd.DataFrame({"c": [1, 2, 3], "d": [2, 3, 4]}, index=idx)
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.groupby(level=level).sum(), gdf.groupby(level=level).sum())


def test_advanced_groupby_levels():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1], "z": [1, 1, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()
    assert_eq(pdg, gdg)
    pdh = pdg.groupby(level=1).sum()
    gdh = gdg.groupby(level=1).sum()
    assert_eq(pdh, gdh)
    pdg = pdf.groupby(["x", "y", "z"]).sum()
    gdg = gdf.groupby(["x", "y", "z"]).sum()
    pdg = pdf.groupby(["z"]).sum()
    gdg = gdf.groupby(["z"]).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["y", "z"]).sum()
    gdg = gdf.groupby(["y", "z"]).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["x", "z"]).sum()
    gdg = gdf.groupby(["x", "z"]).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["y"]).sum()
    gdg = gdf.groupby(["y"]).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["x"]).sum()
    gdg = gdf.groupby(["x"]).sum()
    assert_eq(pdg, gdg)
    pdh = pdg.groupby(level=0).sum()
    gdh = gdg.groupby(level=0).sum()
    assert_eq(pdh, gdh)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()
    pdh = pdg.groupby(level=[0, 1]).sum()
    gdh = gdg.groupby(level=[0, 1]).sum()
    assert_eq(pdh, gdh)
    pdh = pdg.groupby(level=[1, 0]).sum()
    gdh = gdg.groupby(level=[1, 0]).sum()
    assert_eq(pdh, gdh)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()
    with pytest.raises(IndexError) as raises:
        pdh = pdg.groupby(level=2).sum()
    raises.match("Too many levels")
    with pytest.raises(IndexError) as raises:
        gdh = gdg.groupby(level=2).sum()
    # we use a different error message
    raises.match("Invalid level number")
    assert_eq(pdh, gdh)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby(["x", "y", "z"]).sum(),
        lambda df: df.groupby(["x", "y"]).sum(),
        lambda df: df.groupby(["x", "y"]).agg("sum"),
        lambda df: df.groupby(["y"]).sum(),
        lambda df: df.groupby(["y"]).agg("sum"),
        lambda df: df.groupby(["x"]).sum(),
        lambda df: df.groupby(["x"]).agg("sum"),
        lambda df: df.groupby(["x", "y"]).z.sum(),
        lambda df: df.groupby(["x", "y"]).z.agg("sum"),
    ],
)
def test_empty_groupby(func):
    pdf = pd.DataFrame({"x": [], "y": [], "z": []})
    gdf = cudf.from_pandas(pdf)
    assert_eq(func(pdf), func(gdf), check_index_type=False)


def test_groupby_unsupported_columns():
    np.random.seed(12)
    pd_cat = pd.Categorical(
        pd.Series(np.random.choice(["a", "b", 1], 3), dtype="category")
    )
    pdf = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
            "z": ["d", "e", "f"],
            "a": [3, 4, 5],
        }
    )
    pdf["b"] = pd_cat
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby("x").sum()
    gdg = gdf.groupby("x").sum()
    assert_eq(pdg, gdg)


def test_list_of_series():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby([pdf.x]).y.sum()
    gdg = gdf.groupby([gdf.x]).y.sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby([pdf.x, pdf.y]).y.sum()
    gdg = gdf.groupby([gdf.x, gdf.y]).y.sum()
    pytest.skip()
    assert_eq(pdg, gdg)


def test_groupby_use_agg_column_as_index():
    pdf = pd.DataFrame()
    pdf["a"] = [1, 1, 1, 3, 5]
    gdf = cudf.DataFrame()
    gdf["a"] = [1, 1, 1, 3, 5]
    pdg = pdf.groupby("a").agg({"a": "count"})
    gdg = gdf.groupby("a").agg({"a": "count"})
    assert_eq(pdg, gdg, check_dtype=False)


def test_groupby_list_then_string():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [6, 7, 6, 7, 6]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    pdg = pdf.groupby("a", as_index=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    assert_eq(gdg, pdg)


def test_groupby_different_unequal_length_column_aggregations():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    pdg = pdf.groupby("a", as_index=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    assert_eq(pdg, gdg)


def test_groupby_single_var_two_aggs():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg({"b": ["min", "max"]})
    pdg = pdf.groupby("a", as_index=True).agg({"b": ["min", "max"]})
    assert_eq(pdg, gdg)


def test_groupby_double_var_two_aggs():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["a", "b"], as_index=True).agg({"c": ["min", "max"]})
    pdg = pdf.groupby(["a", "b"], as_index=True).agg({"c": ["min", "max"]})
    assert_eq(pdg, gdg)


def test_groupby_apply_basic_agg_single_column():
    gdf = DataFrame()
    gdf["key"] = [0, 0, 1, 1, 2, 2, 0]
    gdf["val"] = [0, 1, 2, 3, 4, 5, 6]
    gdf["mult"] = gdf["key"] * gdf["val"]
    pdf = gdf.to_pandas()

    gdg = gdf.groupby(["key", "val"]).mult.sum()
    pdg = pdf.groupby(["key", "val"]).mult.sum()
    assert_eq(pdg, gdg)


def test_groupby_multi_agg_single_groupby_series():
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby("x").y.agg(["sum", "max"])
    gdg = gdf.groupby("x").y.agg(["sum", "max"])

    assert_eq(pdg, gdg)


def test_groupby_multi_agg_multi_groupby():
    pdf = pd.DataFrame(
        {
            "a": np.random.randint(0, 5, 10),
            "b": np.random.randint(0, 5, 10),
            "c": np.random.randint(0, 5, 10),
            "d": np.random.randint(0, 5, 10),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["a", "b"]).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"]).agg(["sum", "max"])
    assert_eq(pdg, gdg)


def test_groupby_datetime_multi_agg_multi_groupby():
    from datetime import datetime, timedelta

    pdf = pd.DataFrame(
        {
            "a": pd.date_range(
                datetime.now(), datetime.now() + timedelta(9), freq="D"
            ),
            "b": np.random.randint(0, 5, 10),
            "c": np.random.randint(0, 5, 10),
            "d": np.random.randint(0, 5, 10),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["a", "b"]).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"]).agg(["sum", "max"])

    assert_eq(pdg, gdg)


@pytest.mark.parametrize("agg", ["min", "max", "sum", "count", "mean"])
def test_groupby_nulls_basic(agg):
    check_dtype = False if agg == "count" else True

    pdf = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": [1, 2, 1, 2, 1, None]})
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        getattr(pdf.groupby("a"), agg)(),
        getattr(gdf.groupby("a"), agg)(),
        check_dtype=check_dtype,
    )

    pdf = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [1, 2, 1, 2, 1, None],
            "c": [1, 2, 1, None, 1, 2],
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        getattr(pdf.groupby("a"), agg)(),
        getattr(gdf.groupby("a"), agg)(),
        check_dtype=check_dtype,
    )

    pdf = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [1, 2, 1, 2, 1, None],
            "c": [1, 2, None, None, 1, 2],
        }
    )
    gdf = cudf.from_pandas(pdf)

    # TODO: fillna() used here since we don't follow
    # Pandas' null semantics. Should we change it?
    assert_eq(
        getattr(pdf.groupby("a"), agg)().fillna(0),
        getattr(gdf.groupby("a"), agg)().fillna(0),
        check_dtype=check_dtype,
    )


def test_groupby_nulls_in_index():
    pdf = pd.DataFrame({"a": [None, 2, 1, 1], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.groupby("a").sum(), gdf.groupby("a").sum())


def test_groupby_all_nulls_index():
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([None, None, None, None], dtype="object"),
            "b": [1, 2, 3, 4],
        }
    )
    pdf = gdf.to_pandas()
    assert_eq(pdf.groupby("a").sum(), gdf.groupby("a").sum())

    gdf = cudf.DataFrame(
        {"a": cudf.Series([np.nan, np.nan, np.nan, np.nan]), "b": [1, 2, 3, 4]}
    )
    pdf = gdf.to_pandas()
    assert_eq(pdf.groupby("a").sum(), gdf.groupby("a").sum())


def test_groupby_sort():
    pdf = pd.DataFrame({"a": [2, 2, 1, 1], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a", sort=False).sum().sort_index(),
        gdf.groupby("a", sort=False).sum().sort_index(),
    )

    pdf = pd.DataFrame(
        {"c": [-1, 2, 1, 4], "b": [1, 2, 3, 4], "a": [2, 2, 1, 1]}
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby(["c", "b"], sort=False).sum().sort_index(),
        gdf.groupby(["c", "b"], sort=False).sum().to_pandas().sort_index(),
    )


def test_groupby_cat():
    pdf = pd.DataFrame(
        {"a": [1, 1, 2], "b": pd.Series(["b", "b", "a"], dtype="category")}
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.groupby("a").count(), gdf.groupby("a").count(), check_dtype=False
    )


def test_groupby_index_type():
    df = cudf.DataFrame()
    df["string_col"] = ["a", "b", "c"]
    df["counts"] = [1, 2, 3]
    res = df.groupby(by="string_col").counts.sum()
    assert isinstance(res.index, cudf.core.index.StringIndex)


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("q", [0.25, 0.4, 0.5, 0.7, 1])
def test_groupby_quantile(interpolation, q):
    raw_data = {
        "y": [None, 1, 2, 3, 4, None, 6, 7, 8, 9],
        "x": [1, 2, 3, 1, 2, 2, 1, None, 3, 2],
    }
    # Pandas>0.25 now casts NaN in quantile operations as a float64
    # # so we are filling with zeros.
    pdf = pd.DataFrame(raw_data).fillna(0)
    gdf = DataFrame.from_pandas(pdf)

    pdg = pdf.groupby("x")
    gdg = gdf.groupby("x")

    pdresult = pdg.quantile(q, interpolation=interpolation)
    gdresult = gdg.quantile(q, interpolation=interpolation)

    # There's a lot left to add to python bindings like index name
    # so this is a temporary workaround
    pdresult = pdresult["y"].reset_index(drop=True)
    gdresult = gdresult["y"].reset_index(drop=True)

    if q == 0.5 and interpolation == "nearest":
        pytest.xfail(
            "Pandas NaN Rounding will fail nearest interpolation at 0.5"
        )

    assert_eq(pdresult, gdresult)


def test_groupby_std():
    raw_data = {
        "x": [1, 2, 3, 1, 2, 2, 1, None, 3, 2],
        "y": [None, 1, 2, 3, 4, None, 6, 7, 8, 9],
    }
    pdf = pd.DataFrame(raw_data)
    gdf = DataFrame.from_pandas(pdf)
    pdg = pdf.groupby("x")
    gdg = gdf.groupby("x")
    pdresult = pdg.std()
    gdresult = gdg.std()

    # There's a lot left to add to python bindings like index name
    # so this is a temporary workaround
    pdresult = pdresult["y"].reset_index(drop=True)
    gdresult = gdresult["y"].reset_index(drop=True)
    assert_eq(pdresult, gdresult)


def test_groupby_size():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 3, 4],
            "b": ["bob", "bob", "alice", "cooper"],
            "c": [1, 2, 3, 4],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a").size(), gdf.groupby("a").size(), check_dtype=False
    )

    assert_eq(
        pdf.groupby(["a", "b", "c"]).size(),
        gdf.groupby(["a", "b", "c"]).size(),
        check_dtype=False,
    )

    sr = pd.Series(range(len(pdf)))
    assert_eq(
        pdf.groupby(sr).size(), gdf.groupby(sr).size(), check_dtype=False
    )


@pytest.mark.parametrize("nelem", get_nelem())
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("agg", ["min", "max", "mean", "count"])
def test_groupby_datetime(nelem, as_index, agg):
    if agg == "mean" and as_index is True:
        return
    check_dtype = agg not in ("mean", "count")
    pdf = make_frame(pd.DataFrame, nelem=nelem, with_datetime=True)
    gdf = make_frame(cudf.DataFrame, nelem=nelem, with_datetime=True)
    pdg = pdf.groupby("datetime", as_index=as_index)
    gdg = gdf.groupby("datetime", as_index=as_index)
    if as_index is False:
        pdres = getattr(pdg, agg)()
        gdres = getattr(gdg, agg)()
    else:
        pdres = pdg.agg({"datetime": agg})
        gdres = gdg.agg({"datetime": agg})
    assert_eq(pdres, gdres, check_dtype=check_dtype)


def test_groupby_dropna():
    df = cudf.DataFrame({"a": [1, 1, None], "b": [1, 2, 3]})
    expect = cudf.DataFrame(
        {"b": [3, 3]}, index=cudf.Series([1, None], name="a")
    )
    got = df.groupby("a", dropna=False).sum()
    assert_eq(expect, got)

    df = cudf.DataFrame(
        {"a": [1, 1, 1, None], "b": [1, None, 1, None], "c": [1, 2, 3, 4]}
    )
    idx = cudf.MultiIndex.from_frame(
        df[["a", "b"]].drop_duplicates(), names=["a", "b"]
    )
    expect = cudf.DataFrame({"c": [4, 2, 4]}, index=idx)
    got = df.groupby(["a", "b"], dropna=False).sum()

    assert_eq(expect, got)


def test_groupby_dropna_getattr():
    df = cudf.DataFrame()
    df["id"] = [0, 1, 1, None, None, 3, 3]
    df["val"] = [0, 1, 1, 2, 2, 3, 3]
    got = df.groupby("id", dropna=False).val.sum()

    expect = cudf.Series(
        [0, 2, 6, 4], name="val", index=cudf.Series([0, 1, 3, None], name="id")
    )

    assert_eq(expect, got)


def test_groupby_categorical_from_string():
    gdf = cudf.DataFrame()
    gdf["id"] = ["a", "b", "c"]
    gdf["val"] = [0, 1, 2]
    gdf["id"] = gdf["id"].astype("category")
    assert_eq(
        cudf.DataFrame({"val": gdf["val"]}).set_index(index=gdf["id"]),
        gdf.groupby("id").sum(),
    )


def test_groupby_arbitrary_length_series():
    gdf = cudf.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}, index=[4, 5, 6])
    gsr = cudf.Series([1.0, 2.0, 2.0], index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr = gsr.to_pandas()

    expect = pdf.groupby(psr).sum()
    got = gdf.groupby(gsr).sum()

    assert_eq(expect, got)


def test_groupby_series_same_name_as_dataframe_column():
    gdf = cudf.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}, index=[4, 5, 6])
    gsr = cudf.Series([1.0, 2.0, 2.0], name="a", index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr = gsr.to_pandas()

    expect = pdf.groupby(psr).sum()
    got = gdf.groupby(gsr).sum()

    assert_eq(expect, got)


def test_group_by_series_and_column_name_in_by():
    gdf = cudf.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]}, index=[1, 2, 3]
    )
    gsr0 = cudf.Series([0.0, 1.0, 2.0], name="a", index=[1, 2, 3])
    gsr1 = cudf.Series([0.0, 1.0, 3.0], name="b", index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr0 = gsr0.to_pandas()
    psr1 = gsr1.to_pandas()

    expect = pdf.groupby(["x", psr0, psr1]).sum()
    got = gdf.groupby(["x", gsr0, gsr1]).sum()

    assert_eq(expect, got)
