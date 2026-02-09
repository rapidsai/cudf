# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_rangeindex_intersection_default_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for intersection operation.
    idx1 = cudf.RangeIndex(0, 100)
    # Intersecting two RangeIndex will _always_ result in a RangeIndex, use
    # regular index here to force materializing.
    idx2 = cudf.Index([50, 102])

    expected = cudf.Index([50], dtype=f"int{default_integer_bitwidth}")
    actual = idx1.intersection(idx2)

    assert_eq(expected, actual)
