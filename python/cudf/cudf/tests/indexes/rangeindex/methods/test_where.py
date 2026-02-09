# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_rangeindex_where_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for where operation.
    idx = cudf.RangeIndex(0, 10)
    mask = [True, False, True, False, True, False, True, False, True, False]
    actual = idx.where(mask, -1)
    expected = cudf.Index(
        [0, -1, 2, -1, 4, -1, 6, -1, 8, -1],
        dtype=f"int{default_integer_bitwidth}",
    )
    assert_eq(expected, actual)
