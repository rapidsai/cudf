# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

from cudf.core.index import DatetimeIndex


@pytest.mark.parametrize(
    "testlist",
    [
        [
            "2001-01-01 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-04-11 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-03-08 16:00:00",
            "2001-02-03 08:00:00",
            "2001-01-01 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-01-01 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-01-01 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-01-01 00:00:00",
        ],
    ],
)
def test_datetime_index_is_unique_monotonic(testlist):
    index = DatetimeIndex(testlist)
    index_pd = pd.DatetimeIndex(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing
