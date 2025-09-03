# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf


def test_rangeindex_contains():
    ridx = cudf.RangeIndex(start=0, stop=10, name="Index")
    assert 9 in ridx
    assert 10 not in ridx


@pytest.mark.parametrize(
    "start, stop, step", [(10, 20, 1), (0, -10, -1), (5, 5, 1)]
)
def test_range_index_is_unique_monotonic(start, stop, step):
    index = cudf.RangeIndex(start=start, stop=stop, step=step)
    index_pd = pd.RangeIndex(start=start, stop=stop, step=step)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing
