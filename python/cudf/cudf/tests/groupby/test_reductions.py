# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq, assert_groupby_results_equal, assert_neq
from cudf.testing._utils import assert_exceptions_equal


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


def test_advanced_groupby_levels():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1], "z": [1, 1, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdh = pdg.groupby(level=1).sum()
    gdh = gdg.groupby(level=1).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdg = pdf.groupby(["x", "y", "z"]).sum()
    gdg = gdf.groupby(["x", "y", "z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["z"]).sum()
    gdg = gdf.groupby(["z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["y", "z"]).sum()
    gdg = gdf.groupby(["y", "z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["x", "z"]).sum()
    gdg = gdf.groupby(["x", "z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["y"]).sum()
    gdg = gdf.groupby(["y"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["x"]).sum()
    gdg = gdf.groupby(["x"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdh = pdg.groupby(level=0).sum()
    gdh = gdg.groupby(level=0).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()
    pdh = pdg.groupby(level=[0, 1]).sum()
    gdh = gdg.groupby(level=[0, 1]).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdh = pdg.groupby(level=[1, 0]).sum()
    gdh = gdg.groupby(level=[1, 0]).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()

    assert_exceptions_equal(
        lfunc=pdg.groupby,
        rfunc=gdg.groupby,
        lfunc_args_and_kwargs=([], {"level": 2}),
        rfunc_args_and_kwargs=([], {"level": 2}),
    )


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
    assert_groupby_results_equal(func(pdf), func(gdf), check_index_type=False)


def test_groupby_unsupported_columns():
    rng = np.random.default_rng(seed=12)
    pd_cat = pd.Categorical(
        pd.Series(rng.choice(["a", "b", 1], 3), dtype="category")
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
    pdg = pdf.groupby("x").sum(numeric_only=True)
    # cudf does not yet support numeric_only, so our default is False (unlike
    # pandas, which defaults to inferring and throws a warning about it).
    gdg = gdf.groupby("x").sum(numeric_only=True)
    assert_groupby_results_equal(pdg, gdg)


def test_list_of_series():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby([pdf.x]).y.sum()
    gdg = gdf.groupby([gdf.x]).y.sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby([pdf.x, pdf.y]).y.sum()
    gdg = gdf.groupby([gdf.x, gdf.y]).y.sum()
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_apply_basic_agg_single_column():
    gdf = cudf.DataFrame(
        {
            "key": [0, 0, 1, 1, 2, 2, 0],
            "val": [0, 1, 2, 3, 4, 5, 6],
            "mult": [0, 1, 2, 3, 4, 5, 6],
        }
    )
    pdf = gdf.to_pandas()

    gdg = gdf.groupby(["key", "val"]).mult.sum()
    pdg = pdf.groupby(["key", "val"]).mult.sum()
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_nulls_in_index():
    pdf = pd.DataFrame({"a": [None, 2, 1, 1], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").sum(), gdf.groupby("a").sum()
    )


def test_groupby_all_nulls_index():
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([None, None, None, None], dtype="object"),
            "b": [1, 2, 3, 4],
        }
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby("a").sum(), gdf.groupby("a").sum()
    )

    gdf = cudf.DataFrame(
        {"a": cudf.Series([np.nan, np.nan, np.nan, np.nan]), "b": [1, 2, 3, 4]}
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby("a").sum(), gdf.groupby("a").sum()
    )


@pytest.mark.parametrize("sort", [True, False])
def test_groupby_sort(sort):
    pdf = pd.DataFrame({"a": [2, 2, 1, 1], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a", sort=sort).sum(),
        gdf.groupby("a", sort=sort).sum(),
        check_like=not sort,
    )

    pdf = pd.DataFrame(
        {"c": [-1, 2, 1, 4], "b": [1, 2, 3, 4], "a": [2, 2, 1, 1]}
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby(["c", "b"], sort=sort).sum(),
        gdf.groupby(["c", "b"], sort=sort).sum(),
        check_like=not sort,
    )

    ps = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=[2, 2, 2, 3, 3, 1, 1, 1])
    gs = cudf.from_pandas(ps)

    assert_groupby_results_equal(
        ps.groupby(level=0, sort=sort).sum().to_frame(),
        gs.groupby(level=0, sort=sort).sum().to_frame(),
        check_like=not sort,
    )

    ps = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8],
        index=pd.MultiIndex.from_product([(1, 2), ("a", "b"), (42, 84)]),
    )
    gs = cudf.from_pandas(ps)

    assert_groupby_results_equal(
        ps.groupby(level=0, sort=sort).sum().to_frame(),
        gs.groupby(level=0, sort=sort).sum().to_frame(),
        check_like=not sort,
    )


def test_groupby_cat():
    pdf = pd.DataFrame(
        {"a": [1, 1, 2], "b": pd.Series(["b", "b", "a"], dtype="category")}
    )
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        pdf.groupby("a").count(),
        gdf.groupby("a").count(),
        check_dtype=False,
    )


def test_groupby_index_type():
    df = cudf.DataFrame()
    df["string_col"] = ["a", "b", "c"]
    df["counts"] = [1, 2, 3]
    res = df.groupby(by="string_col").counts.sum()
    assert res.index.dtype == cudf.dtype("object")


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("q", [0.25, 0.4, 0.5, 0.7, 1])
def test_groupby_quantile(request, interpolation, q):
    request.applymarker(
        pytest.mark.xfail(
            condition=(q == 0.5 and interpolation == "nearest"),
            reason=(
                "Pandas NaN Rounding will fail nearest interpolation at 0.5"
            ),
        )
    )

    raw_data = {
        "y": [None, 1, 2, 3, 4, None, 6, 7, 8, 9],
        "x": [1, 2, 3, 1, 2, 2, 1, None, 3, 2],
    }
    # Pandas>0.25 now casts NaN in quantile operations as a float64
    # # so we are filling with zeros.
    pdf = pd.DataFrame(raw_data).fillna(0)
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdg = pdf.groupby("x")
    gdg = gdf.groupby("x")

    pdresult = pdg.quantile(q, interpolation=interpolation)
    gdresult = gdg.quantile(q, interpolation=interpolation)

    assert_groupby_results_equal(pdresult, gdresult)


def test_groupby_std():
    raw_data = {
        "x": [1, 2, 3, 1, 2, 2, 1, None, 3, 2],
        "y": [None, 1, 2, 3, 4, None, 6, 7, 8, 9],
    }
    pdf = pd.DataFrame(raw_data)
    gdf = cudf.DataFrame.from_pandas(pdf)
    pdg = pdf.groupby("x")
    gdg = gdf.groupby("x")
    pdresult = pdg.std()
    gdresult = gdg.std()

    assert_groupby_results_equal(pdresult, gdresult)


def test_groupby_size():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 3, 4],
            "b": ["bob", "bob", "alice", "cooper"],
            "c": [1, 2, 3, 4],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").size(),
        gdf.groupby("a").size(),
        check_dtype=False,
    )

    assert_groupby_results_equal(
        pdf.groupby(["a", "b", "c"]).size(),
        gdf.groupby(["a", "b", "c"]).size(),
        check_dtype=False,
    )

    sr = pd.Series(range(len(pdf)))
    assert_groupby_results_equal(
        pdf.groupby(sr).size(),
        gdf.groupby(sr).size(),
        check_dtype=False,
    )


def test_groupby_datetime(request, as_index, groupby_reduction_methods):
    pdf = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "val": [7, 8, 9],
            "datetime": pd.date_range("2020-01-01", periods=3),
        }
    )
    gdf = cudf.DataFrame(pdf)
    pdg = pdf.groupby("datetime", as_index=as_index)
    gdg = gdf.groupby("datetime", as_index=as_index)
    pdres = getattr(pdg, groupby_reduction_methods)()
    gdres = getattr(gdg, groupby_reduction_methods)()
    assert_groupby_results_equal(
        pdres,
        gdres,
        as_index=as_index,
        by=["datetime"],
    )


def test_groupby_dropna():
    df = cudf.DataFrame({"a": [1, 1, None], "b": [1, 2, 3]})
    expect = cudf.DataFrame(
        {"b": [3, 3]}, index=cudf.Series([1, None], name="a")
    )
    got = df.groupby("a", dropna=False).sum()
    assert_groupby_results_equal(expect, got)

    df = cudf.DataFrame(
        {"a": [1, 1, 1, None], "b": [1, None, 1, None], "c": [1, 2, 3, 4]}
    )
    idx = cudf.MultiIndex.from_frame(
        df[["a", "b"]].drop_duplicates().sort_values(["a", "b"]),
        names=["a", "b"],
    )
    expect = cudf.DataFrame({"c": [4, 2, 4]}, index=idx)
    got = df.groupby(["a", "b"], dropna=False).sum()

    assert_groupby_results_equal(expect, got)


def test_groupby_dropna_getattr():
    df = cudf.DataFrame()
    df["id"] = [0, 1, 1, None, None, 3, 3]
    df["val"] = [0, 1, 1, 2, 2, 3, 3]
    got = df.groupby("id", dropna=False).val.sum()

    expect = cudf.Series(
        [0, 2, 6, 4], name="val", index=cudf.Series([0, 1, 3, None], name="id")
    )

    assert_groupby_results_equal(expect, got)


def test_groupby_categorical_from_string():
    gdf = cudf.DataFrame()
    gdf["id"] = ["a", "b", "c"]
    gdf["val"] = [0, 1, 2]
    gdf["id"] = gdf["id"].astype("category")
    assert_groupby_results_equal(
        cudf.DataFrame({"val": gdf["val"]}).set_index(keys=gdf["id"]),
        gdf.groupby("id").sum(),
    )


def test_groupby_arbitrary_length_series():
    gdf = cudf.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}, index=[4, 5, 6])
    gsr = cudf.Series([1.0, 2.0, 2.0], index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr = gsr.to_pandas()

    expect = pdf.groupby(psr).sum()
    got = gdf.groupby(gsr).sum()

    assert_groupby_results_equal(expect, got)


def test_groupby_series_same_name_as_dataframe_column():
    gdf = cudf.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}, index=[4, 5, 6])
    gsr = cudf.Series([1.0, 2.0, 2.0], name="a", index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr = gsr.to_pandas()

    expect = pdf.groupby(psr).sum()
    got = gdf.groupby(gsr).sum()

    assert_groupby_results_equal(expect, got)


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

    assert_groupby_results_equal(expect, got)


def test_raise_data_error():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    gdf = cudf.from_pandas(pdf)

    assert_exceptions_equal(
        pdf.groupby("a").mean,
        gdf.groupby("a").mean,
    )


def test_reset_index_after_empty_groupby():
    # GH #5475
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").sum().reset_index(),
        gdf.groupby("a").sum().reset_index(),
        as_index=False,
        by="a",
    )


def test_groupby_attribute_error():
    err_msg = "Test error message"

    class TestGroupBy(cudf.core.groupby.GroupBy):
        @property
        def _groupby(self):
            raise AttributeError(err_msg)

    a = cudf.DataFrame({"a": [1, 2], "b": [2, 3]})
    gb = TestGroupBy(a, a["a"])

    with pytest.raises(AttributeError, match=err_msg):
        gb.sum()


@pytest.mark.parametrize(
    "pdf",
    [pd.DataFrame(), pd.DataFrame({"a": []}), pd.Series([], dtype="float64")],
)
def test_groupby_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    if isinstance(pdf, pd.DataFrame):
        kwargs = {"check_column_type": False}
    else:
        kwargs = {}
    assert_groupby_results_equal(
        pdf.groupby([]).max(),
        gdf.groupby([]).max(),
        check_dtype=False,
        check_index_type=False,  # Int64 v/s Float64
        **kwargs,
    )


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_week(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-03"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-09"),
                pd.Timestamp("2000-01-02"),
                pd.Timestamp("2000-01-07"),
                pd.Timestamp("2000-01-16"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="1W", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="1W", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_day(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-03"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-09"),
                pd.Timestamp("2000-01-02"),
                pd.Timestamp("2000-01-07"),
                pd.Timestamp("2000-01-16"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="3D", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="3D", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_min(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-01 12:01:00"),
                pd.Timestamp("2000-01-01 12:05:00"),
                pd.Timestamp("2000-01-01 15:30:00"),
                pd.Timestamp("2000-01-02 00:00:00"),
                pd.Timestamp("2000-01-01 23:47:00"),
                pd.Timestamp("2000-01-02 00:05:00"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="1h", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="1h", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_s(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-01 00:00:02"),
                pd.Timestamp("2000-01-01 00:00:07"),
                pd.Timestamp("2000-01-01 00:00:02"),
                pd.Timestamp("2000-01-02 00:00:15"),
                pd.Timestamp("2000-01-01 00:00:05"),
                pd.Timestamp("2000-01-02 00:00:09"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="3s", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="3s", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("index_names", ["a", "b", "c", ["b", "c"]])
def test_groupby_by_index_names(index_names):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["a", "b", "a", "a"], "c": [1, 1, 2, 1]}
    ).set_index(index_names)
    pdf = gdf.to_pandas()

    assert_groupby_results_equal(
        pdf.groupby(index_names).min(), gdf.groupby(index_names).min()
    )


@pytest.mark.parametrize(
    "groups", ["a", "b", "c", ["a", "c"], ["a", "b", "c"]]
)
def test_group_by_pandas_compat(groups):
    with cudf.option_context("mode.pandas_compatible", True):
        df = cudf.DataFrame(
            {
                "a": [1, 3, 2, 3, 3],
                "b": ["x", "a", "y", "z", "a"],
                "c": [10, 13, 11, 12, 12],
            }
        )
        pdf = df.to_pandas()

        assert_eq(pdf.groupby(groups).max(), df.groupby(groups).max())


@pytest.mark.parametrize(
    "groups", ["a", "b", "c", ["a", "c"], ["a", "b", "c"]]
)
@pytest.mark.parametrize("sort", [True, False])
def test_group_by_pandas_sort_order(groups, sort):
    with cudf.option_context("mode.pandas_compatible", True):
        df = cudf.DataFrame(
            {
                "a": [10, 1, 10, 3, 2, 1, 3, 3],
                "b": [5, 6, 7, 1, 2, 3, 4, 9],
                "c": [20, 20, 10, 11, 13, 11, 12, 12],
            }
        )
        pdf = df.to_pandas()

        assert_eq(
            pdf.groupby(groups, sort=sort).sum(),
            df.groupby(groups, sort=sort).sum(),
        )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_group_by_empty_reduction(
    all_supported_types_as_str, groupby_reduction_methods, request
):
    request.applymarker(
        pytest.mark.xfail(
            condition=all_supported_types_as_str == "category"
            and groupby_reduction_methods
            in {"min", "max", "idxmin", "idxmax", "first", "last"},
            reason=f"cuDF doesn't support {groupby_reduction_methods} on {all_supported_types_as_str}",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=all_supported_types_as_str == "str"
            and groupby_reduction_methods in {"idxmin", "idxmax"},
            reason=f"cuDF doesn't support {groupby_reduction_methods} on {all_supported_types_as_str}",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition="int" in all_supported_types_as_str
            and groupby_reduction_methods == "mean",
            reason=f"{all_supported_types_as_str} returns incorrect result type with {groupby_reduction_methods}",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition="timedelta" in all_supported_types_as_str
            and groupby_reduction_methods == "prod",
            raises=RuntimeError,
            reason=f"{all_supported_types_as_str} raises libcudf error with {groupby_reduction_methods}",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition="datetime" in all_supported_types_as_str
            and groupby_reduction_methods in {"mean", "prod", "sum"},
            raises=RuntimeError,
            reason=f"{all_supported_types_as_str} raises libcudf error with {groupby_reduction_methods}",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=all_supported_types_as_str in {"str", "category"}
            and groupby_reduction_methods in {"sum", "prod", "mean"},
            raises=TypeError,
            reason=f"{all_supported_types_as_str} raises TypeError with {groupby_reduction_methods}",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=all_supported_types_as_str == "bool"
            and groupby_reduction_methods in {"sum", "prod", "mean"},
            reason=f"{all_supported_types_as_str} returns incorrect result type with {groupby_reduction_methods}",
        )
    )
    gdf = cudf.DataFrame(
        {"a": [], "b": [], "c": []}, dtype=all_supported_types_as_str
    )
    pdf = gdf.to_pandas()

    gg = gdf.groupby("a")["c"]
    pg = pdf.groupby("a", observed=True)["c"]

    assert_eq(
        getattr(gg, groupby_reduction_methods)(),
        getattr(pg, groupby_reduction_methods)(),
        check_dtype=True,
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warning only given on newer versions.",
)
def test_categorical_grouping_pandas_compatibility():
    gdf = cudf.DataFrame(
        {
            "key": cudf.Series([2, 1, 3, 1, 1], dtype="category"),
            "a": [0, 1, 3, 2, 3],
        }
    )
    pdf = gdf.to_pandas()

    with cudf.option_context("mode.pandas_compatible", True):
        actual = gdf.groupby("key", sort=False).sum()
    with pytest.warns(FutureWarning):
        # observed param deprecation.
        expected = pdf.groupby("key", sort=False).sum()
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "by,data",
    [
        ("a", {"a": [1, 2, 3]}),
        (["a", "id"], {"id": [0, 0, 1], "a": [1, 2, 3]}),
        ("a", {"a": [1, 2, 3], "b": ["A", "B", "C"]}),
        ("id", {"id": [0, 0, 1], "a": [1, 2, 3], "b": ["A", "B", "C"]}),
        (["b", "id"], {"id": [0, 0, 1], "b": ["A", "B", "C"]}),
        ("b", {"b": ["A", "B", "C"]}),
    ],
)
def test_group_by_reduce_numeric_only(by, data, groupby_reduction_methods):
    # Test that simple groupby reductions support numeric_only=True
    if groupby_reduction_methods == "count":
        pytest.skip(
            f"{groupby_reduction_methods} doesn't support numeric_only"
        )
    df = cudf.DataFrame(data)
    expected = getattr(
        df.to_pandas().groupby(by, sort=True), groupby_reduction_methods
    )(numeric_only=True)
    result = getattr(df.groupby(by, sort=True), groupby_reduction_methods)(
        numeric_only=True
    )
    assert_eq(expected, result)


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
    "str_data", [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]]
)
def test_string_groupby_key(str_data):
    num_keys = 2
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame(
        {
            0: pd.Series(str_data, dtype="str"),
            1: pd.Series(str_data, dtype="str"),
            "a": other_data,
        }
    )
    gdf = cudf.DataFrame(
        {
            0: cudf.Series(str_data, dtype="str"),
            1: cudf.Series(str_data, dtype="str"),
            "a": other_data,
        }
    )

    expect = pdf.groupby(list(range(num_keys)), as_index=False).count()
    got = gdf.groupby(list(range(num_keys)), as_index=False).count()

    expect = expect.sort_values([0]).reset_index(drop=True)
    got = got.sort_values([0]).reset_index(drop=True)

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "str_data", [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]]
)
@pytest.mark.parametrize("agg", ["count", "max", "min"])
def test_string_groupby_non_key(str_data, agg):
    num_cols = 2
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame(
        {
            0: pd.Series(str_data, dtype="str"),
            1: pd.Series(str_data, dtype="str"),
            "a": other_data,
        }
    )
    gdf = cudf.DataFrame(
        {
            0: cudf.Series(str_data, dtype="str"),
            1: cudf.Series(str_data, dtype="str"),
            "a": other_data,
        }
    )

    expect = getattr(pdf.groupby("a", as_index=False), agg)()
    got = getattr(gdf.groupby("a", as_index=False), agg)()

    expect = expect.sort_values(["a"]).reset_index(drop=True)
    got = got.sort_values(["a"]).reset_index(drop=True)

    if agg in ["min", "max"] and len(expect) == 0 and len(got) == 0:
        for i in range(num_cols):
            expect[i] = expect[i].astype("str")

    assert_eq(expect, got, check_dtype=False)


def test_string_groupby_key_index():
    str_data = ["a", "b", "c", "d", "e"]
    other_data = [1, 2, 3, 4, 5]

    pdf = pd.DataFrame(
        {
            "a": pd.Series(str_data, dtype="str"),
            "b": other_data,
        }
    )
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series(str_data, dtype="str"),
            "b": other_data,
        }
    )

    expect = pdf.groupby("a", sort=True).count()
    got = gdf.groupby("a", sort=True).count()

    assert_eq(expect, got, check_dtype=False)
