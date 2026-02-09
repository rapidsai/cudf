# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_rangeindex_union_default_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for union operation.
    idx1 = cudf.RangeIndex(0, 2)
    idx2 = cudf.RangeIndex(5, 6)

    expected = cudf.Index([0, 1, 5], dtype=f"int{default_integer_bitwidth}")
    actual = idx1.union(idx2)

    assert_eq(expected, actual)
