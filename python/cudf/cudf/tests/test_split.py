# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data, delimiter, index, expected",
    [
        # Tera original case
        (["a_b_c", "d_e", "f"], "_", 1, ["b", "e", None]),
        # Extra: Index 0
        (["hello|world", "foo"], "|", 0, ["hello", "foo"]),
        # Out of bounds (should return None)
        (["one_two", "three"], "_", 2, [None, None]),
        # Empty, None, and no match
        ([None, "", "x_y_z", "no_delim"], "_", 1, [None, None, "y", None]),
    ],
)
def test_split_part(data, delimiter, index, expected):
    s = cudf.Series(data)
    got = s.str.split_part(delimiter=delimiter, index=index)
    expect = cudf.Series(expected)
    assert_eq(got, expect)
