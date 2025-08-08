# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.groupby.testing import assert_groupby_results_equal


def test_groupby_mean():
    pdf = pd.DataFrame(np.ones((20, 3)), columns=["x", "y", "val"])
    gdf = cudf.DataFrame(pdf)
    got_df = gdf.groupby(["x", "y"]).mean()
    expect_df = pdf.groupby(["x", "y"]).mean()
    assert_groupby_results_equal(got_df, expect_df)


def test_groupby_mean_3level():
    pdf = pd.DataFrame(np.ones((20, 4)), columns=["x", "y", "val", "z"])
    gdf = cudf.DataFrame(pdf)
    bys = list("xyz")
    got_df = pdf.groupby(bys).mean()
    expect_df = gdf.groupby(bys).mean()
    assert_groupby_results_equal(got_df, expect_df)


def test_group_keys_true():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y", group_keys=True).sum()
    pdf = pdf.groupby("y", group_keys=True).sum()
    assert_groupby_results_equal(pdf, gdf)


def test_groupby_getitem_getattr(as_index):
    pdf = pd.DataFrame({"x": [1, 3, 1], "y": [1, 2, 3], "z": [1, 4, 5]})
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        pdf.groupby("x", as_index=as_index)["y"].sum(),
        gdf.groupby("x", as_index=as_index)["y"].sum(),
        as_index=as_index,
        by="x",
    )
    assert_groupby_results_equal(
        pdf.groupby("x", as_index=as_index).y.sum(),
        gdf.groupby("x", as_index=as_index).y.sum(),
        as_index=as_index,
        by="x",
    )
    assert_groupby_results_equal(
        pdf.groupby("x", as_index=as_index)[["y"]].sum(),
        gdf.groupby("x", as_index=as_index)[["y"]].sum(),
        as_index=as_index,
        by="x",
    )
    assert_groupby_results_equal(
        pdf.groupby(["x", "y"], as_index=as_index).sum(),
        gdf.groupby(["x", "y"], as_index=as_index).sum(),
        as_index=as_index,
        by=["x", "y"],
    )


def test_groupby_cats():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {"cats": pd.Categorical(list("aabaacaab")), "vals": rng.random(9)}
    )

    cats = df["cats"].values_host
    vals = df["vals"].to_numpy()

    grouped = df.groupby(["cats"], as_index=False).mean()

    got_vals = grouped["vals"]

    got_cats = grouped["cats"]

    for i in range(len(got_vals)):
        expect = vals[cats == got_cats[i]].mean()
        np.testing.assert_almost_equal(got_vals[i], expect)


def test_series_groupby(groupby_reduction_methods):
    s = pd.Series([1, 2, 3])
    g = cudf.Series([1, 2, 3])
    sg = s.groupby(s // 2)
    gg = g.groupby(g // 2)
    sa = getattr(sg, groupby_reduction_methods)()
    ga = getattr(gg, groupby_reduction_methods)()
    assert_groupby_results_equal(sa, ga)


def test_groupby_level_zero(groupby_reduction_methods, request):
    request.applymarker(
        pytest.mark.xfail(
            groupby_reduction_methods in ["idxmin", "idxmax"],
            reason="gather needed for idxmin/idxmax",
        )
    )
    pdf = pd.DataFrame({"x": [1, 2, 3]}, index=[2, 5, 5])
    gdf = cudf.DataFrame.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, groupby_reduction_methods)()
    gdresult = getattr(gdg, groupby_reduction_methods)()
    assert_groupby_results_equal(
        pdresult,
        gdresult,
    )


def test_groupby_series_level_zero(groupby_reduction_methods, request):
    request.applymarker(
        pytest.mark.xfail(
            groupby_reduction_methods in ["idxmin", "idxmax"],
            reason="gather needed for idxmin/idxmax",
        )
    )
    pdf = pd.Series([1, 2, 3], index=[2, 5, 5])
    gdf = cudf.Series.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, groupby_reduction_methods)()
    gdresult = getattr(gdg, groupby_reduction_methods)()
    assert_groupby_results_equal(pdresult, gdresult)


def test_groupby_column_name():
    pdf = pd.DataFrame({"xx": [1.0, 2.0, 3.0], "yy": [1, 2, 3]})
    gdf = cudf.DataFrame.from_pandas(pdf)
    g = gdf.groupby("yy")
    p = pdf.groupby("yy")
    gxx = g["xx"].sum()
    pxx = p["xx"].sum()
    assert_groupby_results_equal(pxx, gxx)

    gxx = g["xx"].count()
    pxx = p["xx"].count()
    assert_groupby_results_equal(pxx, gxx, check_dtype=False)

    gxx = g["xx"].min()
    pxx = p["xx"].min()
    assert_groupby_results_equal(pxx, gxx)

    gxx = g["xx"].max()
    pxx = p["xx"].max()
    assert_groupby_results_equal(pxx, gxx)

    gxx = g["xx"].idxmin()
    pxx = p["xx"].idxmin()
    assert_groupby_results_equal(pxx, gxx, check_dtype=False)

    gxx = g["xx"].idxmax()
    pxx = p["xx"].idxmax()
    assert_groupby_results_equal(pxx, gxx, check_dtype=False)

    gxx = g["xx"].mean()
    pxx = p["xx"].mean()
    assert_groupby_results_equal(pxx, gxx)


def test_groupby_column_numeral():
    pdf = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [1, 2, 3]})
    gdf = cudf.DataFrame.from_pandas(pdf)
    p = pdf.groupby(1)
    g = gdf.groupby(1)
    pxx = p[0].sum()
    gxx = g[0].sum()
    assert_groupby_results_equal(pxx, gxx)

    pdf = pd.DataFrame({0.5: [1.0, 2.0, 3.0], 1.5: [1, 2, 3]})
    gdf = cudf.DataFrame.from_pandas(pdf)
    p = pdf.groupby(1.5)
    g = gdf.groupby(1.5)
    pxx = p[0.5].sum()
    gxx = g[0.5].sum()
    assert_groupby_results_equal(pxx, gxx)


@pytest.mark.parametrize(
    "series",
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [1, 2, 3],
        [4, 3, 2],
        [0, 2, 0],
        pd.Series([0, 2, 0]),
        pd.Series([0, 2, 0], index=[0, 2, 1]),
    ],
)
def test_groupby_external_series(series):
    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]})
    gdf = cudf.DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_groupby_results_equal(pxx, gxx)


@pytest.mark.parametrize("series", [[0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
def test_groupby_external_series_incorrect_length(series):
    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]})
    gdf = cudf.DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_groupby_results_equal(pxx, gxx)


@pytest.mark.parametrize(
    "level", [0, 1, "a", "b", [0, 1], ["a", "b"], ["a", 1], -1, [-1, -2]]
)
def test_groupby_levels(level):
    idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 2)], names=("a", "b"))
    pdf = pd.DataFrame({"c": [1, 2, 3], "d": [2, 3, 4]}, index=idx)
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        pdf.groupby(level=level).sum(),
        gdf.groupby(level=level).sum(),
    )
