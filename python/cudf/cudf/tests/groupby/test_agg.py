# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import decimal

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.tests.groupby.testing import assert_groupby_results_equal


@pytest.mark.parametrize("empty", [True, False])
def test_agg_count_dtype(empty):
    df = cudf.DataFrame({"a": [1, 2, 1], "c": ["a", "b", "c"]})
    if empty:
        df = df.iloc[:0]
    result = df.groupby("a").agg({"c": "count"})
    assert result["c"].dtype == np.dtype("int64")


@pytest.mark.parametrize("attr", ["agg", "aggregate"])
def test_series_agg(attr):
    df = cudf.DataFrame({"a": [1, 2, 1, 2], "b": [0, 0, 0, 0]})
    pdf = df.to_pandas()
    agg = getattr(df.groupby("a")["a"], attr)("count")
    pd_agg = getattr(pdf.groupby(["a"])["a"], attr)("count")

    assert agg.ndim == pd_agg.ndim


@pytest.mark.parametrize("func", ["sum", "prod", "mean", "count"])
@pytest.mark.parametrize("attr", ["agg", "aggregate"])
def test_dataframe_agg(attr, func):
    df = cudf.DataFrame({"a": [1, 2, 1, 2], "b": [0, 0, 0, 0]})
    pdf = df.to_pandas()

    agg = getattr(df.groupby("a"), attr)(func)
    pd_agg = getattr(pdf.groupby(["a"]), attr)(func)

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)({"b": func})
    pd_agg = getattr(pdf.groupby(["a"]), attr)({"b": func})

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)([func])
    pd_agg = getattr(pdf.groupby(["a"]), attr)([func])

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)(foo=("b", func), bar=("a", func))
    pd_agg = getattr(pdf.groupby(["a"]), attr)(
        foo=("b", func), bar=("a", func)
    )

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)(
        foo=cudf.NamedAgg(column="b", aggfunc=func),
        bar=cudf.NamedAgg(column="a", aggfunc=func),
    )
    pd_agg = getattr(pdf.groupby(["a"]), attr)(
        foo=("b", func), bar=("a", func)
    )

    assert_eq(agg, pd_agg)


def test_dataframe_agg_with_invalid_kwarg():
    with pytest.raises(TypeError, match="Invalid keyword argument"):
        df = cudf.DataFrame({"a": [1, 2, 1, 2], "b": [0, 0, 0, 0]})
        df.groupby("a").agg(foo=set())


@pytest.mark.parametrize("with_nulls", [False, True])
def test_groupby_agg_maintain_order_random(with_nulls):
    nrows = 20
    nkeys = 3
    rng = np.random.default_rng(seed=0)
    key_names = [f"key{key}" for key in range(nkeys)]
    key_values = [rng.integers(100, size=nrows) for _ in key_names]
    value = rng.integers(-100, 100, size=nrows)
    df = cudf.DataFrame(
        dict(zip(key_names, key_values, strict=True), value=value)
    )
    if with_nulls:
        for key in key_names:
            df.loc[df[key] == 1, key] = None
    with cudf.option_context("mode.pandas_compatible", True):
        got = df.groupby(key_names, sort=False).agg({"value": "sum"})
    expect = (
        df.to_pandas().groupby(key_names, sort=False).agg({"value": "sum"})
    )
    assert_eq(expect, got, check_index_type=not with_nulls)


def test_groupby_agg_mean_min():
    pdf = pd.DataFrame(np.ones((20, 3)), columns=["x", "y", "val"])
    gdf = cudf.DataFrame(pdf)
    got_df = gdf.groupby(["x", "y"]).agg(["mean", "min"])
    expect_df = pdf.groupby(["x", "y"]).agg(["mean", "min"])
    assert_groupby_results_equal(got_df, expect_df)


def test_groupby_agg_min_max_dictargs():
    pdf = pd.DataFrame(np.ones((20, 5)), columns=["x", "y", "val", "a", "b"])
    gdf = cudf.DataFrame(pdf)
    expect_df = pdf.groupby(["x", "y"]).agg({"a": "min", "b": "max"})
    got_df = gdf.groupby(["x", "y"]).agg({"a": "min", "b": "max"})
    assert_groupby_results_equal(expect_df, got_df)


def test_groupby_agg_min_max_dictlist():
    pdf = pd.DataFrame(np.ones((20, 5)), columns=["x", "y", "val", "a", "b"])
    gdf = cudf.DataFrame(pdf)
    expect_df = pdf.groupby(["x", "y"]).agg(
        {"a": ["min", "max"], "b": ["min", "max"]}
    )
    got_df = gdf.groupby(["x", "y"]).agg(
        {"a": ["min", "max"], "b": ["min", "max"]}
    )
    assert_groupby_results_equal(got_df, expect_df)


def test_groupby_as_index_single_agg(as_index):
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    pdf = pdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    assert_groupby_results_equal(pdf, gdf, as_index=as_index, by="y")


def test_groupby_default():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y").agg({"x": "mean"})
    pdf = pdf.groupby("y").agg({"x": "mean"})
    assert_groupby_results_equal(pdf, gdf)


def test_groupby_as_index_multiindex(as_index):
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
        for gcol, pcol in zip(gdf, pdf, strict=True):
            np.testing.assert_array_equal(
                gdf[gcol].to_numpy(), pdf[pcol].values
            )


@pytest.mark.parametrize(
    "func",
    [
        "mean",
        "std",
        "var",
        "min",
        "max",
        "idxmin",
        "idxmax",
        "count",
        "sum",
        "prod",
    ],
)
def test_groupby_2keys_agg(func):
    # gdf (Note: lack of multiIndex)
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 2)), columns=["x", "y"])
    gdf = cudf.DataFrame(pdf)
    expect_df = pdf.groupby(["x", "y"]).agg(func)
    got_df = gdf.groupby(["x", "y"]).agg(func)

    assert_groupby_results_equal(got_df, expect_df)


def test_series_groupby_agg(groupby_reduction_methods):
    s = pd.Series([1, 2, 3])
    g = cudf.Series([1, 2, 3])
    sg = s.groupby(s // 2).agg(groupby_reduction_methods)
    gg = g.groupby(g // 2).agg(groupby_reduction_methods)
    assert_groupby_results_equal(sg, gg)


def test_groupby_agg_decimal(groupby_reduction_methods, request):
    request.applymarker(
        pytest.mark.xfail(
            groupby_reduction_methods in ["prod", "mean"],
            raises=pd.errors.DataError,
            reason=f"{groupby_reduction_methods} not supported with Decimals in pandas",
        )
    )
    rng = np.random.default_rng(seed=0)
    num_groups = 4
    nelem_per_group = 10
    # The number of digits after the decimal to use.
    decimal_digits = 2
    # The number of digits before the decimal to use.
    whole_digits = 2

    scale = 10**whole_digits
    nelem = num_groups * nelem_per_group

    # The unique is necessary because otherwise if there are duplicates idxmin
    # and idxmax may return different results than pandas (see
    # https://github.com/rapidsai/cudf/issues/7756). This is not relevant to
    # the current version of the test, because idxmin and idxmax simply don't
    # work with pandas Series composed of Decimal objects (see
    # https://github.com/pandas-dev/pandas/issues/40685). However, if that is
    # ever enabled, then this issue will crop up again so we may as well have
    # it fixed now.
    x = np.unique((rng.random(nelem) * scale).round(decimal_digits))
    y = np.unique((rng.random(nelem) * scale).round(decimal_digits))

    if x.size < y.size:
        total_elements = x.size
        y = y[: x.size]
    else:
        total_elements = y.size
        x = x[: y.size]

    # Note that this filtering can lead to one group with fewer elements, but
    # that shouldn't be a problem and is probably useful to test.
    idx_col = np.tile(np.arange(num_groups), nelem_per_group)[:total_elements]

    decimal_x = pd.Series([decimal.Decimal(str(d)) for d in x])
    decimal_y = pd.Series([decimal.Decimal(str(d)) for d in y])

    pdf = pd.DataFrame({"idx": idx_col, "x": decimal_x, "y": decimal_y})
    gdf = cudf.DataFrame(
        {
            "idx": idx_col,
            "x": cudf.Series(decimal_x),
            "y": cudf.Series(decimal_y),
        }
    )

    expect_df = pdf.groupby("idx", sort=True).agg(groupby_reduction_methods)
    got_df = gdf.groupby("idx", sort=True).agg(groupby_reduction_methods)
    assert_eq(expect_df["x"], got_df["x"], check_dtype=False)
    assert_eq(expect_df["y"], got_df["y"], check_dtype=False)
