# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series([1, 2, 3, 4, 5, 10, 11, 12, 33, 55, 19]),
        pd.Series(["abc", "def", "ghi", "xyz", "pqr", "abc"]),
        pd.Series(
            [1, 2, 3, 4, 5, 10],
            index=["abc", "def", "ghi", "xyz", "pqr", "abc"],
        ),
        pd.Series(
            ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            index=[1, 2, 3, 4, 5, 10],
        ),
        pd.Series(index=["a", "b", "c", "d", "e", "f"], dtype="float64"),
        pd.Series(index=[10, 11, 12], dtype="float64"),
        pd.Series(dtype="float64"),
        pd.Series([], dtype="float64"),
    ],
)
def test_series_keys(ps):
    gds = cudf.from_pandas(ps)

    assert_eq(ps.keys(), gds.keys())
