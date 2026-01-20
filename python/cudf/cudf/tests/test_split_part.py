# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

# SPDX-License-Identifier: Apache-2.0

import cudf
from cudf.testing._utils import assert_eq


def test_split_part():
    s = cudf.Series(["a_b_c", "d_e", "f"])

    # Case 1: Index 1
    got = s.str.split_part(delimiter="_", index=1)
    expect = cudf.Series(["b", "e", None])
    assert_eq(got, expect)

    # Case 2: Index 0
    got = s.str.split_part(delimiter="_", index=0)
    expect = cudf.Series(["a", "d", "f"])
    assert_eq(got, expect)

def test_split_part_whitespace():
    # Case: Whitespace splitting (spaces, tabs, mixed)
    s = cudf.Series(["a b c", "d  e", "f\tg", " h "])
    
    # delimiter="" should act like whitespace split
    # Index 0
    got = s.str.split_part(delimiter="", index=0)
    expect = cudf.Series(["a", "d", "f", "h"])
    assert_eq(got, expect)

    # Index 1
    got = s.str.split_part(delimiter="", index=1)
    expect = cudf.Series(["b", "e", "g", None])
    assert_eq(got, expect)