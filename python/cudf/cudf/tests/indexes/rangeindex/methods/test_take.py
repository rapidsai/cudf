# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_rangeindex_take_default_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for take operation.
    idx = cudf.RangeIndex(0, 100)
    actual = idx.take([0, 3, 7, 62])
    expected = cudf.Index(
        [0, 3, 7, 62], dtype=f"int{default_integer_bitwidth}"
    )
    assert_eq(expected, actual)
