# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        pd.Index([1, 10, 2, 100, -10], name="abc"),
        pd.Index(["z", "x", "a", "c", "b"]),
        pd.Index(["z", "x", "a", "c", "b"], dtype="category"),
        pd.Index(
            [-10.2, 100.1, -100.2, 0.0, 0.23], name="this is a float index"
        ),
        pd.Index([102, 1001, 1002, 0.0, 23], dtype="datetime64[ns]"),
        pd.Index([13240.2, 1001, 100.2, 0.0, 23], dtype="datetime64[ns]"),
        pd.RangeIndex(0, 10, 1),
        pd.RangeIndex(0, -100, -2),
        pd.Index([-10.2, 100.1, -100.2, 0.0, 23], dtype="timedelta64[ns]"),
    ],
)
@pytest.mark.parametrize("return_indexer", [True, False])
def test_index_sort_values(data, ascending, return_indexer):
    pdi = data
    gdi = cudf.from_pandas(pdi)

    expected = pdi.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )
    actual = gdi.sort_values(
        ascending=ascending, return_indexer=return_indexer
    )

    if return_indexer:
        expected_indexer = expected[1]
        actual_indexer = actual[1]

        assert_eq(expected_indexer, actual_indexer)

        expected = expected[0]
        actual = actual[0]

    assert_eq(expected, actual)
