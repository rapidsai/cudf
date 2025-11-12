# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "testlist", [["c", "d", "e", "f"], ["z", "y", "x", "r"]]
)
def test_categorical_index_is_unique_monotonic(testlist):
    # Assuming unordered categorical data cannot be "monotonic"
    raw_cat = pd.Categorical(testlist, ordered=True)
    index = cudf.CategoricalIndex(raw_cat)
    index_pd = pd.CategoricalIndex(raw_cat)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "data", [["$ 1", "$ 2", "hello"], ["($) 1", "( 2", "hello", "^1$"]]
)
@pytest.mark.parametrize("value", ["$ 1", "hello", "$", "^1$"])
def test_categorical_string_index_contains(data, value):
    idx = cudf.CategoricalIndex(data)
    pidx = idx.to_pandas()

    assert_eq(value in idx, value in pidx)


@pytest.mark.parametrize("ordered", [True, False])
def test_index_ordered(ordered):
    pd_ci = pd.CategoricalIndex([1, 2, 3], ordered=ordered)
    cudf_ci = cudf.from_pandas(pd_ci)
    assert pd_ci.ordered == cudf_ci.ordered


def test_categoricalindex_constructor():
    gidx = cudf.CategoricalIndex(["a", "b", "c"])

    assert gidx._constructor is cudf.CategoricalIndex


@pytest.mark.parametrize(
    "data",
    [
        ["a", "b", "c"],
        [1, 2, 3],
        pd.Categorical(["a", "b", "c"]),
    ],
)
def test_categoricalindex_inferred_type(data):
    gidx = cudf.CategoricalIndex(data)
    pidx = pd.CategoricalIndex(data)
    assert_eq(gidx.inferred_type, pidx.inferred_type)
