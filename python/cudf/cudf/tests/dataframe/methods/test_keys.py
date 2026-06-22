# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3, 4, 5, 10, 11, 12, 33, 55, 19]}),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            }
        ),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            },
            index=[10, 20, 30, 40, 50, 60],
        ),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            },
            index=["a", "b", "c", "d", "e", "f"],
        ),
        pd.DataFrame(index=["a", "b", "c", "d", "e", "f"]),
        pd.DataFrame(columns=["a", "b", "c", "d", "e", "f"]),
        pd.DataFrame(index=[10, 11, 12]),
        pd.DataFrame(columns=[10, 11, 12]),
        pd.DataFrame(),
        pd.DataFrame({"one": [], "two": []}),
        pd.DataFrame({2: [], 1: []}),
        pd.DataFrame(
            {
                0: [1, 2, 3, 4, 5, 10],
                1: ["abc", "def", "ghi", "xyz", "pqr", "abc"],
                100: ["a", "b", "b", "x", "z", "a"],
            },
            index=[10, 20, 30, 40, 50, 60],
        ),
    ],
)
def test_dataframe_keys(df):
    gdf = cudf.from_pandas(df)

    assert_eq(df.keys(), gdf.keys())
