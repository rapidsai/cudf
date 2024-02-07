# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import numpy as np
import pytest

import cudf


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
