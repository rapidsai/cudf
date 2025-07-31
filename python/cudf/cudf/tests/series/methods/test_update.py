# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("index", [None, [1, 2, 3]])
@pytest.mark.parametrize(
    "other",
    [
        pd.Series([4, 5, 6]),
        pd.Series([4, 5, 6, 7, 8]),
        pd.Series([4, np.nan, 6]),
        [4, np.nan, 6],
        {1: 9},
    ],
)
def test_series_update(index, other):
    pd_data = pd.Series([1, 2, 3], index=index)
    data = cudf.Series.from_pandas(pd_data)
    gs = data.copy(deep=True)
    if isinstance(other, pd.Series):
        other = cudf.Series.from_pandas(other, nan_as_null=False)
        g_other = other.copy(deep=True)
        p_other = g_other.to_pandas()
    else:
        g_other = other
        p_other = other

    ps = gs.to_pandas()

    ps.update(p_other)
    gs.update(g_other)
    assert_eq(gs, ps)
