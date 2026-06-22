# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "gdi, fill_value, expected",
    [
        (
            lambda: cudf.MultiIndex(
                levels=[[1, 3, 4, None], [1, 2, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            5,
            lambda: cudf.MultiIndex(
                levels=[[1, 3, 4, 5], [1, 2, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
        ),
        (
            lambda: cudf.MultiIndex(
                levels=[[1, 3, 4, None], [1, None, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            100,
            lambda: cudf.MultiIndex(
                levels=[[1, 3, 4, 100], [1, 100, 5]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
        ),
        (
            lambda: cudf.MultiIndex(
                levels=[["a", "b", "c", None], ["1", None, "5"]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            "100",
            lambda: cudf.MultiIndex(
                levels=[["a", "b", "c", "100"], ["1", "100", "5"]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
        ),
    ],
)
def test_multiindex_fillna(gdi, fill_value, expected):
    assert_eq(expected(), gdi().fillna(fill_value))
