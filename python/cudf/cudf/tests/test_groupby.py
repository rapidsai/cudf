# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import itertools

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import cudf
from cudf.core import DataFrame, Series
from cudf.core._compat import PANDAS_GE_110
from cudf.tests.utils import assert_eq, assert_exceptions_equal

_now = np.datetime64("now")
_tomorrow = _now + np.timedelta64(1, "D")
_now = np.int64(_now.astype("datetime64[ns]"))
_tomorrow = np.int64(_tomorrow.astype("datetime64[ns]"))
_index_type_aggs = {"count", "idxmin", "idxmax"}


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


def get_nelem():
    for elem in [2, 3, 1000]:
        yield elem


@pytest.fixture
def gdf():
    return DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})


@pytest.fixture
def pdf(gdf):
    return gdf.to_pandas()


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_mean(nelem):
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .mean()
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .mean()
    )
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_mean_3level(nelem):
    lvls = "z"
    bys = list("xyz")
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_levels=lvls)
        .groupby(bys, sort=True)
        .mean()
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_levels=lvls)
        .groupby(bys, sort=True)
        .mean()
    )
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_agg_mean_min(nelem):
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .agg(["mean", "min"])
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .agg(["mean", "min"])
    )
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictargs(nelem):
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"], sort=True)
        .agg({"a": "min", "b": "max"})
    )
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"], sort=True)
        .agg({"a": "min", "b": "max"})
    )
    assert_eq(expect_df, got_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictlist(nelem):
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"], sort=True)
        .agg({"a": ["min", "max"], "b": ["min", "max"]})
    )
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"], sort=True)
        .agg({"a": ["min", "max"], "b": ["min", "max"]})
    )
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
@pytest.mark.parametrize(
    "func", ["mean", "min", "max", "idxmin", "idxmax", "count", "sum"]
)
def test_groupby_2keys_agg(nelem, func):
    # gdf (Note: lack of multiIndex)
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .agg(func)
    )
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .agg(func)
    )
    check_dtype = False if func in _index_type_aggs else True
    assert_eq(got_df, expect_df, check_dtype=check_dtype)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_as_index_single_agg(pdf, gdf, as_index):
    gdf = gdf.groupby("y", as_index=as_index, sort=True).agg({"x": "mean"})
    pdf = pdf.groupby("y", as_index=as_index, sort=True).agg({"x": "mean"})
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_as_index_multiindex(pdf, gdf, as_index):
    pdf = pd.DataFrame(
        {"a": [1, 2, 1], "b": [3, 3, 3], "c": [2, 2, 3], "d": [3, 1, 2]}
    )
    gdf = cudf.from_pandas(pdf)

    gdf = gdf.groupby(["a", "b"], as_index=as_index, sort=True).agg(
        {"c": "mean"}
    )
    pdf = pdf.groupby(["a", "b"], as_index=as_index, sort=True).agg(
        {"c": "mean"}
    )

    if as_index:
        assert_eq(pdf, gdf)
    else:
        # column names don't match - check just the values
        for gcol, pcol in zip(gdf, pdf):
            assert_array_equal(gdf[gcol].to_array(), pdf[pcol].values)


def test_groupby_default(pdf, gdf):
    gdf = gdf.groupby("y", sort=True).agg({"x": "mean"})
    pdf = pdf.groupby("y", sort=True).agg({"x": "mean"})
    assert_eq(pdf, gdf)


def test_group_keys_true(pdf, gdf):
    gdf = gdf.groupby("y", group_keys=True, sort=True).sum()
    pdf = pdf.groupby("y", group_keys=True, sort=True).sum()
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_getitem_getattr(as_index):
    pdf = pd.DataFrame({"x": [1, 3, 1], "y": [1, 2, 3], "z": [1, 4, 5]})
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.groupby("x", sort=True)["y"].sum(),
        gdf.groupby("x", sort=True)["y"].sum(),
    )
    assert_eq(
        pdf.groupby("x", sort=True).y.sum(),
        gdf.groupby("x", sort=True).y.sum(),
    )
    assert_eq(
        pdf.groupby("x", sort=True)[["y"]].sum(),
        gdf.groupby("x", sort=True)[["y"]].sum(),
    )
    assert_eq(
        pdf.groupby(["x", "y"], as_index=as_index, sort=True).sum(),
        gdf.groupby(["x", "y"], as_index=as_index, sort=True).sum(),
    )


def test_groupby_cats():
    df = DataFrame()
    df["cats"] = pd.Categorical(list("aabaacaab"))
    df["vals"] = np.random.random(len(df))

    cats = df["cats"].values_host
    vals = df["vals"].to_array()

    grouped = df.groupby(["cats"], as_index=False).mean()

    got_vals = grouped["vals"]

    got_cats = grouped["cats"]

    for i in range(len(got_vals)):
        expect = vals[cats == got_cats[i]].mean()
        np.testing.assert_almost_equal(got_vals[i], expect)


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

    for name, grp in df.groupby(["key1", "key2"]):
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

    expect_grpby = df.to_pandas().groupby(
        ["key1", "key2"], as_index=False, sort=True
    )
    got_grpby = df.groupby(["key1", "key2"], sort=True)

    def foo(df):
        df["out"] = df["val1"] + df["val2"]
        return df

    expect = expect_grpby.apply(foo)
    got = got_grpby.apply(foo)
    assert_eq(expect, got)


def test_groupby_apply_grouped():
    from numba import cuda

    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df["key1"] = np.random.randint(0, 3, nelem)
    df["key2"] = np.random.randint(0, 2, nelem)
    df["val1"] = np.random.random(nelem)
    df["val2"] = np.random.random(nelem)

    expect_grpby = df.to_pandas().groupby(
        ["key1", "key2"], as_index=False, sort=True
    )
    got_grpby = df.groupby(["key1", "key2"], sort=True)

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
    expect = expect.sort_values(["key1", "key2"])

    assert_eq(expect, got)


@pytest.mark.parametrize("nelem", [100, 500])
@pytest.mark.parametrize(
    "func",
    ["mean", "std", "var", "min", "max", "idxmin", "idxmax", "count", "sum"],
)
def test_groupby_cudf_2keys_agg(nelem, func):
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .agg(func)
    )

    # pandas
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .agg(func)
    )
    check_dtype = False if func in _index_type_aggs else True
    assert_eq(got_df, expect_df, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmin", "idxmax", "count", "sum", "mean"]
)
def test_series_groupby(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2)
    gg = g.groupby(g // 2)
    sa = getattr(sg, agg)()
    ga = getattr(gg, agg)()
    check_dtype = False if agg in _index_type_aggs else True
    assert_eq(sa, ga, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmin", "idxmax", "count", "sum", "mean"]
)
def test_series_groupby_agg(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2).agg(agg)
    gg = g.groupby(g // 2).agg(agg)
    check_dtype = False if agg in _index_type_aggs else True
    assert_eq(sg, gg, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg",
    [
        "min",
        "max",
        "count",
        "sum",
        "mean",
        pytest.param(
            "idxmin",
            marks=pytest.mark.xfail(reason="gather needed for idxmin"),
        ),
        pytest.param(
            "idxmax",
            marks=pytest.mark.xfail(reason="gather needed for idxmax"),
        ),
    ],
)
def test_groupby_level_zero(agg):
    pdf = pd.DataFrame({"x": [1, 2, 3]}, index=[2, 5, 5])
    gdf = DataFrame.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    check_dtype = False if agg in _index_type_aggs else True
    assert_eq(pdresult, gdresult, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg",
    [
        "min",
        "max",
        "count",
        "sum",
        "mean",
        pytest.param(
            "idxmin",
            marks=pytest.mark.xfail(reason="gather needed for idxmin"),
        ),
        pytest.param(
            "idxmax",
            marks=pytest.mark.xfail(reason="gather needed for idxmax"),
        ),
    ],
)
def test_groupby_series_level_zero(agg):
    pdf = pd.Series([1, 2, 3], index=[2, 5, 5])
    gdf = Series.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    check_dtype = False if agg in _index_type_aggs else True
    assert_eq(pdresult, gdresult, check_dtype=check_dtype)


def test_groupby_column_name():
    pdf = pd.DataFrame({"xx": [1.0, 2.0, 3.0], "yy": [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    g = gdf.groupby("yy", sort=True)
    p = pdf.groupby("yy", sort=True)
    gxx = g["xx"].sum()
    pxx = p["xx"].sum()
    assert_eq(pxx, gxx)

    gxx = g["xx"].count()
    pxx = p["xx"].count()
    assert_eq(pxx, gxx, check_dtype=False)

    gxx = g["xx"].min()
    pxx = p["xx"].min()
    assert_eq(pxx, gxx)

    gxx = g["xx"].max()
    pxx = p["xx"].max()
    assert_eq(pxx, gxx)

    gxx = g["xx"].idxmin()
    pxx = p["xx"].idxmin()
    assert_eq(pxx, gxx, check_dtype=False)

    gxx = g["xx"].idxmax()
    pxx = p["xx"].idxmax()
    assert_eq(pxx, gxx, check_dtype=False)

    gxx = g["xx"].mean()
    pxx = p["xx"].mean()
    assert_eq(pxx, gxx)


def test_groupby_column_numeral():
    pdf = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    p = pdf.groupby(1, sort=True)
    g = gdf.groupby(1, sort=True)
    pxx = p[0].sum()
    gxx = g[0].sum()
    assert_eq(pxx, gxx)

    pdf = pd.DataFrame({0.5: [1.0, 2.0, 3.0], 1.5: [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    p = pdf.groupby(1.5, sort=True)
    g = gdf.groupby(1.5, sort=True)
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
    pxx = pdf.groupby(pd.Series(series), sort=True).x.sum()
    gxx = gdf.groupby(cudf.Series(series), sort=True).x.sum()
    assert_eq(pxx, gxx)


@pytest.mark.parametrize("series", [[0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
def test_groupby_external_series_incorrect_length(series):
    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]})
    gdf = DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series), sort=True).x.sum()
    gxx = gdf.groupby(cudf.Series(series), sort=True).x.sum()
    assert_eq(pxx, gxx)


@pytest.mark.parametrize(
    "level", [0, 1, "a", "b", [0, 1], ["a", "b"], ["a", 1], -1, [-1, -2]]
)
def test_groupby_levels(level):
    idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 2)], names=("a", "b"))
    pdf = pd.DataFrame({"c": [1, 2, 3], "d": [2, 3, 4]}, index=idx)
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.groupby(level=level, sort=True).sum(),
        gdf.groupby(level=level, sort=True).sum(),
    )


def test_advanced_groupby_levels():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1], "z": [1, 1, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["x", "y"], sort=True).sum()
    gdg = gdf.groupby(["x", "y"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdh = pdg.groupby(level=1, sort=True).sum()
    gdh = gdg.groupby(level=1, sort=True).sum()
    assert_eq(pdh, gdh)
    pdg = pdf.groupby(["x", "y", "z"], sort=True).sum()
    gdg = gdf.groupby(["x", "y", "z"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["z"], sort=True).sum()
    gdg = gdf.groupby(["z"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["y", "z"], sort=True).sum()
    gdg = gdf.groupby(["y", "z"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["x", "z"], sort=True).sum()
    gdg = gdf.groupby(["x", "z"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["y"], sort=True).sum()
    gdg = gdf.groupby(["y"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(["x"], sort=True).sum()
    gdg = gdf.groupby(["x"], sort=True).sum()
    assert_eq(pdg, gdg)
    pdh = pdg.groupby(level=0, sort=True).sum()
    gdh = gdg.groupby(level=0, sort=True).sum()
    assert_eq(pdh, gdh)
    pdg = pdf.groupby(["x", "y"], sort=True).sum()
    gdg = gdf.groupby(["x", "y"], sort=True).sum()
    pdh = pdg.groupby(level=[0, 1], sort=True).sum()
    gdh = gdg.groupby(level=[0, 1], sort=True).sum()
    assert_eq(pdh, gdh)
    pdh = pdg.groupby(level=[1, 0], sort=True).sum()
    gdh = gdg.groupby(level=[1, 0], sort=True).sum()
    assert_eq(pdh, gdh)
    pdg = pdf.groupby(["x", "y"], sort=True).sum()
    gdg = gdf.groupby(["x", "y"], sort=True).sum()

    assert_exceptions_equal(
        lfunc=pdg.groupby,
        rfunc=gdg.groupby,
        lfunc_args_and_kwargs=([], {"level": 2}),
        rfunc_args_and_kwargs=([], {"level": 2}),
        expected_error_message="Invalid level number",
    )


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(
            lambda df: df.groupby(["x", "y", "z"]).sum(),
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/32464"
            ),
        ),
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
    pdg = pdf.groupby("x", sort=True).sum()
    gdg = gdf.groupby("x", sort=True).sum()
    assert_eq(pdg, gdg)


def test_list_of_series():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby([pdf.x], sort=True).y.sum()
    gdg = gdf.groupby([gdf.x], sort=True).y.sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby([pdf.x, pdf.y], sort=True).y.sum()
    gdg = gdf.groupby([gdf.x, gdf.y], sort=True).y.sum()
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
    gdg = gdf.groupby("a", as_index=True, sort=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    pdg = pdf.groupby("a", as_index=True, sort=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    assert_eq(gdg, pdg)


def test_groupby_different_unequal_length_column_aggregations():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True, sort=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    pdg = pdf.groupby("a", as_index=True, sort=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    assert_eq(pdg, gdg)


def test_groupby_single_var_two_aggs():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True, sort=True).agg({"b": ["min", "max"]})
    pdg = pdf.groupby("a", as_index=True, sort=True).agg({"b": ["min", "max"]})
    assert_eq(pdg, gdg)


def test_groupby_double_var_two_aggs():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["a", "b"], as_index=True, sort=True).agg(
        {"c": ["min", "max"]}
    )
    pdg = pdf.groupby(["a", "b"], as_index=True, sort=True).agg(
        {"c": ["min", "max"]}
    )
    assert_eq(pdg, gdg)


def test_groupby_apply_basic_agg_single_column():
    gdf = DataFrame()
    gdf["key"] = [0, 0, 1, 1, 2, 2, 0]
    gdf["val"] = [0, 1, 2, 3, 4, 5, 6]
    gdf["mult"] = gdf["key"] * gdf["val"]
    pdf = gdf.to_pandas()

    gdg = gdf.groupby(["key", "val"], sort=True).mult.sum()
    pdg = pdf.groupby(["key", "val"], sort=True).mult.sum()
    assert_eq(pdg, gdg)


def test_groupby_multi_agg_single_groupby_series():
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby("x", sort=True).y.agg(["sum", "max"])
    gdg = gdf.groupby("x", sort=True).y.agg(["sum", "max"])

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
    pdg = pdf.groupby(["a", "b"], sort=True).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"], sort=True).agg(["sum", "max"])
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
    pdg = pdf.groupby(["a", "b"], sort=True).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"], sort=True).agg(["sum", "max"])

    assert_eq(pdg, gdg)


@pytest.mark.parametrize(
    "agg",
    [
        ["min", "max", "count", "mean"],
        ["mean", "var", "std"],
        ["count", "mean", "var", "std"],
    ],
)
def test_groupby_multi_agg_hash_groupby(agg):
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    prefixes = alphabets[:10]
    coll_dict = dict()
    for prefix in prefixes:
        for this_name in alphabets:
            coll_dict[prefix + this_name] = float
    coll_dict["id"] = int
    gdf = cudf.datasets.timeseries(
        start="2000", end="2000-01-2", dtypes=coll_dict, freq="1s", seed=1,
    ).reset_index(drop=True)
    pdf = gdf.to_pandas()
    check_dtype = False if "count" in agg else True
    pdg = pdf.groupby("id", sort=True).agg(agg)
    gdg = gdf.groupby("id", sort=True).agg(agg)
    assert_eq(pdg, gdg, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmax", "idxmax", "sum", "count", "mean"]
)
def test_groupby_nulls_basic(agg):
    check_dtype = False if agg in _index_type_aggs else True

    pdf = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": [1, 2, 1, 2, 1, None]})
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        getattr(pdf.groupby("a", sort=True), agg)(),
        getattr(gdf.groupby("a", sort=True), agg)(),
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
        getattr(pdf.groupby("a", sort=True), agg)(),
        getattr(gdf.groupby("a", sort=True), agg)(),
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
        getattr(pdf.groupby("a", sort=True), agg)().fillna(0),
        getattr(gdf.groupby("a", sort=True), agg)().fillna(0),
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


@pytest.mark.parametrize("sort", [True, False])
def test_groupby_sort(sort):
    pdf = pd.DataFrame({"a": [2, 2, 1, 1], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a", sort=sort).sum(),
        gdf.groupby("a", sort=sort).sum(),
        check_like=not sort,
    )

    pdf = pd.DataFrame(
        {"c": [-1, 2, 1, 4], "b": [1, 2, 3, 4], "a": [2, 2, 1, 1]}
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby(["c", "b"], sort=sort).sum(),
        gdf.groupby(["c", "b"], sort=sort).sum(),
        check_like=not sort,
    )

    ps = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=[2, 2, 2, 3, 3, 1, 1, 1])
    gs = cudf.from_pandas(ps)

    assert_eq(
        ps.groupby(level=0, sort=sort).sum().to_frame(),
        gs.groupby(level=0, sort=sort).sum().to_frame(),
        check_like=not sort,
    )

    ps = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8],
        index=pd.MultiIndex.from_product([(1, 2), ("a", "b"), (42, 84)]),
    )
    gs = cudf.from_pandas(ps)

    assert_eq(
        ps.groupby(level=0, sort=sort).sum().to_frame(),
        gs.groupby(level=0, sort=sort).sum().to_frame(),
        check_like=not sort,
    )


def test_groupby_cat():
    pdf = pd.DataFrame(
        {"a": [1, 1, 2], "b": pd.Series(["b", "b", "a"], dtype="category")}
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.groupby("a", sort=True).count(),
        gdf.groupby("a", sort=True).count(),
        check_dtype=False,
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
    pdg = pdf.groupby("x", sort=True)
    gdg = gdf.groupby("x", sort=True)
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
        pdf.groupby("a", sort=True).size(),
        gdf.groupby("a", sort=True).size(),
        check_dtype=False,
    )

    assert_eq(
        pdf.groupby(["a", "b", "c"], sort=True).size(),
        gdf.groupby(["a", "b", "c"], sort=True).size(),
        check_dtype=False,
    )

    sr = pd.Series(range(len(pdf)))
    assert_eq(
        pdf.groupby(sr, sort=True).size(),
        gdf.groupby(sr, sort=True).size(),
        check_dtype=False,
    )


@pytest.mark.parametrize("nelem", get_nelem())
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmin", "idxmax", "mean", "count"]
)
def test_groupby_datetime(nelem, as_index, agg):
    if agg == "mean" and as_index is True:
        return
    check_dtype = agg not in ("mean", "count", "idxmin", "idxmax")
    pdf = make_frame(pd.DataFrame, nelem=nelem, with_datetime=True)
    gdf = make_frame(cudf.DataFrame, nelem=nelem, with_datetime=True)
    pdg = pdf.groupby("datetime", as_index=as_index, sort=True)
    gdg = gdf.groupby("datetime", as_index=as_index, sort=True)
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
    got = df.groupby("a", dropna=False, sort=True).sum()
    assert_eq(expect, got)

    df = cudf.DataFrame(
        {"a": [1, 1, 1, None], "b": [1, None, 1, None], "c": [1, 2, 3, 4]}
    )
    idx = cudf.MultiIndex.from_frame(
        df[["a", "b"]].drop_duplicates().sort_values(["a", "b"]),
        names=["a", "b"],
    )
    expect = cudf.DataFrame({"c": [4, 2, 4]}, index=idx)
    got = df.groupby(["a", "b"], dropna=False, sort=True).sum()

    assert_eq(expect, got)


def test_groupby_dropna_getattr():
    df = cudf.DataFrame()
    df["id"] = [0, 1, 1, None, None, 3, 3]
    df["val"] = [0, 1, 1, 2, 2, 3, 3]
    got = df.groupby("id", dropna=False, sort=True).val.sum()

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
        cudf.DataFrame({"val": gdf["val"]}).set_index(keys=gdf["id"]),
        gdf.groupby("id", sort=True).sum(),
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


@pytest.mark.parametrize(
    "grouper",
    [
        "a",
        ["a"],
        ["a", "b"],
        np.array([0, 1, 1, 2, 3, 2]),
        {0: "a", 1: "a", 2: "b", 3: "a", 4: "b", 5: "c"},
        lambda x: x + 1,
        ["a", np.array([0, 1, 1, 2, 3, 2])],
    ],
)
def test_grouping(grouper):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 1, 2, 1, 2],
            "c": [1, 2, 3, 4, 5, 6],
        }
    )
    gdf = cudf.from_pandas(pdf)

    for pdf_group, gdf_group in zip(
        pdf.groupby(grouper), gdf.groupby(grouper)
    ):
        assert pdf_group[0] == gdf_group[0]
        assert_eq(pdf_group[1], gdf_group[1])


@pytest.mark.parametrize("agg", [lambda x: x.count(), "count"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_count(agg, by):

    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by, sort=True).agg(agg)
    got = gdf.groupby(by, sort=True).agg(agg)

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("agg", [lambda x: x.median(), "median"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_median(agg, by):
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).agg(agg)
    got = gdf.groupby(by).agg(agg)

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("agg", [lambda x: x.nunique(), "nunique"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_nunique(agg, by):
    if not PANDAS_GE_110:
        pytest.xfail("pandas >= 1.1 required")
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).nunique()
    got = gdf.groupby(by).nunique()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "n", [0, 1, 2, 10],
)
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_nth(n, by):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [1, 2, 2, 2, 1],
            "c": [1, 2, None, 4, 5],
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by, sort=True).nth(n)
    got = gdf.groupby(by, sort=True).nth(n)

    assert_eq(expect, got, check_dtype=False)


def test_raise_data_error():

    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    gdf = cudf.from_pandas(pdf)

    assert_exceptions_equal(pdf.groupby("a").mean, gdf.groupby("a").mean)


def test_drop_unsupported_multi_agg():

    gdf = cudf.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": ["a", "b", "c", "d"]}
    )
    assert_eq(
        gdf.groupby("a").agg(["count", "mean"]),
        gdf.groupby("a").agg({"b": ["count", "mean"], "c": ["count"]}),
    )


@pytest.mark.parametrize(
    "agg",
    (
        list(itertools.combinations(["count", "max", "min", "nunique"], 2))
        + [
            {"b": "min", "c": "mean"},
            {"b": "max", "c": "mean"},
            {"b": "count", "c": "mean"},
            {"b": "nunique", "c": "mean"},
        ]
    ),
)
def test_groupby_agg_combinations(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "b": ["a", "a", "b", "c", "d"],
            "c": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a", sort=True).agg(agg),
        gdf.groupby("a", sort=True).agg(agg),
        check_dtype=False,
    )


def test_groupby_apply_noempty_group():
    pdf = pd.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "c": [1, 2, 3, 4]}
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.groupby("a")
        .apply(lambda x: x.iloc[[0, 1]])
        .reset_index(drop=True),
        gdf.groupby("a")
        .apply(lambda x: x.iloc[[0, 1]])
        .reset_index(drop=True),
    )


def test_reset_index_after_empty_groupby():
    # GH #5475
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a", sort=True).sum().reset_index(),
        gdf.groupby("a", sort=True).sum().reset_index(),
    )


def test_groupby_attribute_error():
    err_msg = "Test error message"

    class TestGroupBy(cudf.core.groupby.GroupBy):
        @property
        def _groupby(self):
            raise AttributeError("Test error message")

    a = cudf.DataFrame({"a": [1, 2], "b": [2, 3]})
    gb = TestGroupBy(a, a["a"])

    with pytest.raises(AttributeError, match=err_msg):
        gb.sum()


@pytest.mark.parametrize(
    "by",
    [
        "a",
        "b",
        ["a"],
        ["b"],
        ["a", "b"],
        ["b", "a"],
        np.array([0, 0, 0, 1, 1, 1, 2]),
    ],
)
def test_groupby_groups(by):
    pdf = pd.DataFrame(
        {"a": [1, 2, 1, 2, 1, 2, 3], "b": [1, 2, 3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    pdg = pdf.groupby(by)
    gdg = gdf.groupby(by)

    for key in pdg.groups:
        assert key in gdg.groups
        assert_eq(pdg.groups[key], gdg.groups[key])


@pytest.mark.parametrize(
    "by",
    [
        "a",
        "b",
        ["a"],
        ["b"],
        ["a", "b"],
        ["b", "a"],
        ["a", "c"],
        ["a", "b", "c"],
    ],
)
def test_groupby_groups_multi(by):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 1, 2, 3],
            "b": ["a", "b", "a", "b", "b", "c", "c"],
            "c": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    gdf = cudf.from_pandas(pdf)

    pdg = pdf.groupby(by)
    gdg = gdf.groupby(by)

    for key in pdg.groups:
        assert key in gdg.groups
        assert_eq(pdg.groups[key], gdg.groups[key])


def test_groupby_nunique_series():
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 1, 2]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a")["b"].nunique(),
        gdf.groupby("a")["b"].nunique(),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_simple(list_agg):
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, None, 4, 5, 6]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_of_lists(list_agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [[1, 2], [3, None, 5], None, [], [7, 8], [9]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_single_element(list_agg):
    pdf = pd.DataFrame({"a": [1, 2], "b": [3, None]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "agg", [list, [list, "count"], {"b": list, "c": "sum"}]
)
def test_groupby_list_strings(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": ["b", "a", None, "e", "d"],
            "c": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a").agg(agg),
        gdf.groupby("a").agg(agg),
        check_dtype=False,
    )


def test_groupby_list_columns_excluded():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
            "c": [[1, 2], [3, 4], [5, 6], [7, 8]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a").mean(), gdf.groupby("a").mean(), check_dtype=False
    )

    assert_eq(
        pdf.groupby("a").agg("mean"),
        gdf.groupby("a").agg("mean"),
        check_dtype=False,
    )


def test_groupby_pipe():
    pdf = pd.DataFrame({"A": "a b a b".split(), "B": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("A").pipe(lambda x: x.max() - x.min())
    actual = gdf.groupby("A").pipe(lambda x: x.max() - x.min())

    assert_eq(expected, actual)


def test_groupby_apply_return_scalars():
    pdf = pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "B": [
                0.01,
                np.nan,
                0.03,
                0.04,
                np.nan,
                0.06,
                0.07,
                0.08,
                0.09,
                1.0,
            ],
        }
    )
    gdf = cudf.from_pandas(pdf)

    def custom_map_func(x):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = ticker / 10
        return full

    expected = pdf.groupby("A").apply(lambda x: custom_map_func(x))
    actual = gdf.groupby("A").apply(lambda x: custom_map_func(x))

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "cust_func",
    [lambda x: x - x.max(), lambda x: x.min() - x.max(), lambda x: x.min()],
)
def test_groupby_apply_return_series_dataframe(cust_func):
    pdf = pd.DataFrame(
        {"key": [0, 0, 1, 1, 2, 2, 2], "val": [0, 1, 2, 3, 4, 5, 6]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby(["key"]).apply(cust_func)
    actual = gdf.groupby(["key"]).apply(cust_func)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pdf", [pd.DataFrame(), pd.DataFrame({"a": []}), pd.Series([])]
)
def test_groupby_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.groupby([]).max(),
        gdf.groupby([]).max(),
        check_dtype=False,
        check_index_type=False,  # Int64Index v/s Float64Index
    )


@pytest.mark.parametrize(
    "pdf", [pd.DataFrame(), pd.DataFrame({"a": []}), pd.Series([])]
)
def test_groupby_apply_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.groupby([]).apply(lambda x: x.max()),
        gdf.groupby([]).apply(lambda x: x.max()),
    )


@pytest.mark.parametrize(
    "pdf",
    [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2], "b": [2, 3]})],
)
def test_groupby_nonempty_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lambda: pdf.groupby([]),
        lambda: gdf.groupby([]),
        compare_error_message=False,
    )
