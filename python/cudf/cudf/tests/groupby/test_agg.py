# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
