# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from cudf.core.index import RangeIndex


@pytest.mark.parametrize(
    "start, stop", [(0, 10), (0, 1), (3, 4), (0, 0), (3, 3)]
)
@pytest.mark.parametrize("idx", [-1, 0, 5, 10, 11])
@pytest.mark.parametrize("side", ["left", "right"])
def test_rangeindex_get_slice_bound_basic(start, stop, idx, side):
    pd_index = pd.RangeIndex(start, stop)
    cudf_index = RangeIndex(start, stop)
    expect = pd_index.get_slice_bound(idx, side)
    got = cudf_index.get_slice_bound(idx, side)
    assert expect == got


@pytest.mark.parametrize(
    "start, stop, step",
    [(3, 20, 5), (20, 3, -5), (20, 3, 5), (3, 20, -5), (0, 0, 2), (3, 3, 2)],
)
@pytest.mark.parametrize(
    "label",
    [3, 8, 13, 18, 20, 15, 10, 5, -1, 0, 19, 21, 6, 11, 17],
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_rangeindex_get_slice_bound_step(start, stop, step, label, side):
    pd_index = pd.RangeIndex(start, stop, step)
    cudf_index = RangeIndex(start, stop, step)

    expect = pd_index.get_slice_bound(label, side)
    got = cudf_index.get_slice_bound(label, side)
    assert expect == got
