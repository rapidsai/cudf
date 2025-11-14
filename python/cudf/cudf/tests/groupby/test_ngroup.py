# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "by",
    [
        lambda: "a",
        lambda: "b",
        lambda: ["a", "b"],
        lambda: "c",
        lambda: pd.Series([1, 2, 1, 2, 1, 2]),
        lambda: pd.Series(["x", "y", "y", "x", "z", "x"]),
    ],
)
def test_groupby_ngroup(by, ascending):
    df_ngroup = cudf.DataFrame(
        {
            "a": [2, 2, 1, 1, 2, 3],
            "b": [1, 2, 1, 2, 1, 2],
            "c": ["a", "a", "b", "c", "d", "c"],
        },
        index=[1, 3, 5, 7, 4, 2],
    )
    df_ngroup.index.name = "foo"
    by = by()
    expected = df_ngroup.to_pandas().groupby(by).ngroup(ascending=ascending)
    actual = df_ngroup.groupby(by).ngroup(ascending=ascending)
    assert_eq(expected, actual, check_dtype=False)
