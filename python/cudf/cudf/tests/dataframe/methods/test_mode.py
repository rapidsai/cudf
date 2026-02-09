# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "df",
    [
        lambda: cudf.DataFrame({"a": [1, 2, 3]}),
        lambda: cudf.DataFrame(
            {"a": [1, 2, 3], "b": ["a", "z", "c"]}, index=["a", "z", "x"]
        ),
        lambda: cudf.DataFrame(
            {
                "a": [1, 2, 3, None, 2, 1, None],
                "b": ["a", "z", "c", "a", "v", "z", "z"],
            }
        ),
        lambda: cudf.DataFrame({"a": [], "b": []}),
        lambda: cudf.DataFrame({"a": [None, None], "b": [None, None]}),
        lambda: cudf.DataFrame(
            {
                "a": ["hello", "world", "rapids", "ai", "nvidia"],
                "b": cudf.Series(
                    [1, 21, 21, 11, 11],
                    dtype="timedelta64[s]",
                    index=["a", "b", "c", "d", " e"],
                ),
            },
            index=["a", "b", "c", "d", " e"],
        ),
        lambda: cudf.DataFrame(
            {
                "a": ["hello", None, "world", "rapids", None, "ai", "nvidia"],
                "b": cudf.Series(
                    [1, 21, None, 11, None, 11, None], dtype="datetime64[s]"
                ),
            }
        ),
    ],
)
def test_dataframe_mode(df, numeric_only, dropna):
    df = df()
    pdf = df.to_pandas()

    expected = pdf.mode(numeric_only=numeric_only, dropna=dropna)
    actual = df.mode(numeric_only=numeric_only, dropna=dropna)
    if len(actual.columns) == 0:
        # pandas < 3.0 returns an Index[object] instead of RangeIndex
        actual.columns = expected.columns
    assert_eq(expected, actual, check_dtype=False)
