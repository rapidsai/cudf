# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "empty",
    [True, False],
    ids=["empty", "nonempty"],
)
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
