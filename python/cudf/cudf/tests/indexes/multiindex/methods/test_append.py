# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
            names=("number", "color"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], ["yellow", "violet", "pink", "white"]],
            names=("number1", "color2"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
            names=("number", "color"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], ["yellow", "violet", "pink", "white"]],
            names=("number1", "color2"),
        ),
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
        ),
    ],
)
def test_multiindex_append(data, other):
    pdi = data
    other_pd = other

    gdi = cudf.from_pandas(data)
    other_gd = cudf.from_pandas(other)

    expected = pdi.append(other_pd)
    actual = gdi.append(other_gd)

    assert_eq(expected, actual)


def test_multiindex_append_index_returns_object_index():
    pdi = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    other_pd = pd.Index([3, 4])

    gdi = cudf.from_pandas(pdi)
    other_gd = cudf.from_pandas(other_pd)

    expected = pdi.append(other_pd)
    actual = gdi.append(other_gd)

    assert_eq(expected, actual)


def test_multiindex_append_index_list_returns_object_index():
    pdi = pd.MultiIndex.from_tuples([(0, "a")])
    other_pd = [pd.Index([1, 2]), pd.Index([3, 4])]

    gdi = cudf.from_pandas(pdi)
    other_gd = [cudf.from_pandas(other) for other in other_pd]

    expected = pdi.append(other_pd)
    actual = gdi.append(other_gd)
    assert_eq(expected, actual)
