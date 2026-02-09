# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "gs",
    [
        lambda: cudf.Series([1, 2, 3]),
        lambda: cudf.Series([None]),
        lambda: cudf.Series([4]),
        lambda: cudf.Series([2, 3, -1, 0, 1], name="test name"),
        lambda: cudf.Series(
            [1, 2, 3, None, 2, 1], index=["a", "v", "d", "e", "f", "g"]
        ),
        lambda: cudf.Series([1, 2, 3, None, 2, 1, None], name="abc"),
        lambda: cudf.Series(["ab", "bc", "ab", None, "bc", None, None]),
        lambda: cudf.Series([None, None, None, None, None], dtype="str"),
        lambda: cudf.Series([None, None, None, None, None]),
        lambda: cudf.Series(
            [
                123213,
                23123,
                123123,
                12213123,
                12213123,
                12213123,
                23123,
                2312323123,
                None,
                None,
            ],
            dtype="timedelta64[ns]",
        ),
        lambda: cudf.Series(
            [
                None,
                1,
                2,
                3242434,
                3233243,
                1,
                2,
                1023,
                None,
                12213123,
                None,
                2312323123,
                None,
                None,
            ],
            dtype="datetime64[ns]",
        ),
        lambda: cudf.Series(name="empty series", dtype="float64"),
        lambda: cudf.Series(
            ["a", "b", "c", " ", "a", "b", "z"], dtype="category"
        ),
    ],
)
def test_series_mode(gs, dropna):
    gs = gs()
    ps = gs.to_pandas()

    expected = ps.mode(dropna=dropna)
    actual = gs.mode(dropna=dropna)

    assert_eq(expected, actual, check_dtype=False)
