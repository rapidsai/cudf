# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_multiindex_union_error():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    pidx = midx.to_pandas()

    assert_exceptions_equal(
        midx.union,
        pidx.union,
        lfunc_args_and_kwargs=(["a"],),
        rfunc_args_and_kwargs=(["b"],),
    )


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (
            pd.MultiIndex.from_arrays(
                [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
            ),
            pd.MultiIndex.from_arrays(
                [[1, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], ["Red", "Blue", "Red", "Blue"]],
                names=["a", "b"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], ["Red", "Green", "Red", "Green"]],
                names=["x", "y"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], [0.2, 0.4, 1.4, 10], [3, 3, 2, 4]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
            ),
        ),
    ],
)
@pytest.mark.parametrize("sort", [None, False])
def test_intersection_mulitIndex(idx1, idx2, sort):
    expected = idx1.intersection(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1)
    idx2 = cudf.from_pandas(idx2)

    actual = idx1.intersection(idx2, sort=sort)
    assert_eq(expected, actual, exact=False)


def test_difference():
    midx = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        names=["x", "y"],
    )
    midx2 = cudf.MultiIndex(
        levels=[[1, 3, 4, 5], [1, 2, 5]],
        codes=[[0, 0, 1, 2, 3, 3], [0, 2, 1, 1, 0, 2]],
        names=["x", "y"],
    )

    expected = midx2.to_pandas().difference(midx.to_pandas())
    actual = midx2.difference(midx)
    assert isinstance(actual, cudf.MultiIndex)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (
            pd.MultiIndex.from_arrays(
                [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], ["Red", "Blue", "Red", "Blue"]],
                names=["a", "b"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], ["Red", "Green", "Red", "Green"]],
                names=["x", "y"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            pd.MultiIndex.from_arrays(
                [[3, 3, 2, 4], [0.2, 0.4, 1.4, 10], [3, 3, 2, 4]]
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
                names=["a", "b", "c"],
            ),
            [(2, 6, 12)],
        ),
    ],
)
@pytest.mark.parametrize("sort", [None, False])
def test_union_mulitIndex(idx1, idx2, sort):
    expected = idx1.union(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.MultiIndex) else idx1
    idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.MultiIndex) else idx2

    actual = idx1.union(idx2, sort=sort)
    assert_eq(expected, actual)
