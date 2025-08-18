# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2]},
        {"a": [1, 2, 3], "b": [3, 4, 5]},
        {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6], "c": [1, 3, 5, 7]},
        {"a": [np.nan, 2, 3, 4], "b": [3, 4, np.nan, 6], "c": [1, 3, 5, 7]},
        {1: [1, 2, 3], 2: [3, 4, 5]},
        {"a": [1, None, None], "b": [3, np.nan, np.nan]},
        {1: ["a", "b", "c"], 2: ["q", "w", "u"]},
        {1: ["a", np.nan, "c"], 2: ["q", None, "u"]},
        {},
        {1: [], 2: [], 3: []},
        [1, 2, 3],
    ],
)
def test_axes(data):
    csr = cudf.DataFrame(data)
    psr = pd.DataFrame(data)

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a, exact=False)


def test_iter():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)
    assert list(pdf) == list(gdf)


def test_column_assignment():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    new_cols = ["q", "r", "s"]
    gdf.columns = new_cols
    assert list(gdf.columns) == new_cols


def test_ndim():
    pdf = pd.DataFrame({"x": range(5), "y": range(5, 10)})
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert pdf.ndim == gdf.ndim


@pytest.mark.parametrize(
    "index",
    [
        ["a", "b", "c", "d", "e"],
        np.array(["a", "b", "c", "d", "e"]),
        pd.Index(["a", "b", "c", "d", "e"], name="name"),
    ],
)
def test_string_index(index):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 5)))
    gdf = cudf.DataFrame.from_pandas(pdf)
    pdf.index = index
    gdf.index = index
    assert_eq(pdf, gdf)
