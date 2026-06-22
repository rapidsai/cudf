# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_rangeindex_slice_attr_name():
    start, stop = 0, 10
    rg = cudf.RangeIndex(start, stop, name="myindex")
    sliced_rg = rg[0:9]
    assert rg.name == sliced_rg.name


@pytest.mark.parametrize(
    "start,stop,step",
    [(1, 10, 1), (1, 10, 3), (10, -17, -1), (10, -17, -3)],
)
def test_index_rangeindex_get_item_basic(start, stop, step):
    pridx = pd.RangeIndex(start, stop, step)
    gridx = cudf.RangeIndex(start, stop, step)

    for i in range(-len(pridx), len(pridx)):
        assert pridx[i] == gridx[i]


@pytest.mark.parametrize("start,stop,step", [(1, 10, 3), (10, 1, -3)])
def test_index_rangeindex_get_item_out_of_bounds(start, stop, step):
    gridx = cudf.RangeIndex(start, stop, step)
    with pytest.raises(IndexError):
        gridx[4]


@pytest.mark.parametrize("start,stop,step", [(10, 1, 1), (-17, 10, -3)])
def test_index_rangeindex_get_item_null_range(start, stop, step):
    gridx = cudf.RangeIndex(start, stop, step)

    with pytest.raises(IndexError):
        gridx[0]


@pytest.mark.parametrize(
    "start,stop,step",
    [(-17, 21, 2), (21, -17, -3), (0, 0, 1), (0, 1, -3), (10, 0, 5)],
)
@pytest.mark.parametrize(
    "sl",
    [
        slice(1, 7, 1),
        slice(1, 7, 2),
        slice(-1, 7, 1),
        slice(-1, 7, 2),
        slice(-3, 7, 2),
        slice(7, 1, -2),
        slice(7, -3, -2),
        slice(None, None, 1),
        slice(0, None, 2),
        slice(0, None, 3),
        slice(0, 0, 3),
    ],
)
def test_index_rangeindex_get_item_slices(start, stop, step, sl):
    pridx = pd.RangeIndex(start, stop, step)
    gridx = cudf.RangeIndex(start, stop, step)

    assert_eq(pridx[sl], gridx[sl])


def test_rangeindex_apply_boolean_mask_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for apply boolean mask operation.
    idx = cudf.RangeIndex(0, 8)
    mask = [True, True, True, False, False, False, True, False]
    actual = idx[mask]
    expected = cudf.Index([0, 1, 2, 6], dtype=f"int{default_integer_bitwidth}")
    assert_eq(expected, actual)


def test_df_slice_empty_index():
    idx = cudf.RangeIndex(0)
    assert isinstance(idx[:1], cudf.RangeIndex)
    with pytest.raises(IndexError):
        idx[1]
