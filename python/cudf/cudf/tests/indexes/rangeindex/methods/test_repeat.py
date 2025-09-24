# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf
from cudf.testing import assert_eq


def test_rangeindex_repeat_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for repeat operation.
    idx = cudf.RangeIndex(0, 3)
    actual = idx.repeat(3)
    expected = cudf.Index(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=f"int{default_integer_bitwidth}"
    )
    assert_eq(expected, actual)
