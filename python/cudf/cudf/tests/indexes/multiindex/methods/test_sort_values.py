# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("return_indexer", [True, False])
@pytest.mark.parametrize(
    "pmidx",
    [
        pd.MultiIndex(
            levels=[[1, 3, 4, 5], [1, 2, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex.from_product(
            [["bar", "baz", "foo", "qux"], ["one", "two"]],
            names=["first", "second"],
        ),
        pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=["one", "two", "three"],
        ),
        pd.MultiIndex.from_tuples([(1, 2), (3, 4)]),
    ],
)
def test_multiindex_sort_values(pmidx, ascending, return_indexer):
    pmidx = pmidx
    midx = cudf.from_pandas(pmidx)

    expected = pmidx.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )
    actual = midx.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )

    if return_indexer:
        expected_indexer = expected[1]
        actual_indexer = actual[1]

        assert_eq(expected_indexer, actual_indexer)

        expected = expected[0]
        actual = actual[0]

    assert_eq(expected, actual)
