# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd

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
