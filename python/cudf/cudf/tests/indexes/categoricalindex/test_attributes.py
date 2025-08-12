# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

from cudf.core.index import CategoricalIndex


@pytest.mark.parametrize(
    "testlist", [["c", "d", "e", "f"], ["z", "y", "x", "r"]]
)
def test_categorical_index_is_unique_monotonic(testlist):
    # Assuming unordered categorical data cannot be "monotonic"
    raw_cat = pd.Categorical(testlist, ordered=True)
    index = CategoricalIndex(raw_cat)
    index_pd = pd.CategoricalIndex(raw_cat)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing
