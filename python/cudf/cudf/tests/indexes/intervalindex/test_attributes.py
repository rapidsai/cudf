# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_intervalindex_constructor():
    gidx = cudf.IntervalIndex.from_breaks([0, 1, 2, 3])

    assert gidx._constructor is cudf.IntervalIndex


@pytest.mark.parametrize(
    "closed",
    ["left", "right", "both", "neither"],
)
def test_intervalindex_inferred_type(closed):
    gidx = cudf.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)
    pidx = pd.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)
    assert_eq(gidx.inferred_type, pidx.inferred_type)


@pytest.mark.parametrize(
    "closed",
    ["left", "right", "both", "neither"],
)
def test_intervalindex_closed_left_right(closed):
    """Test closed_left and closed_right properties."""
    gidx = cudf.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)
    pidx = pd.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)

    assert gidx.closed_left == pidx.closed_left
    assert gidx.closed_right == pidx.closed_right
