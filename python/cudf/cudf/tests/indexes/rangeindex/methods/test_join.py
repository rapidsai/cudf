# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_rangeindex_join_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for join.
    idx1 = cudf.RangeIndex(0, 10, name="a")
    idx2 = cudf.RangeIndex(5, 15, name="b")

    actual = idx1.join(idx2, how="inner", sort=True)
    expected = idx1.to_pandas().join(idx2.to_pandas(), how="inner", sort=True)
    assert actual.dtype == cudf.dtype(f"int{default_integer_bitwidth}")
    # exact=False to ignore dtype comparison,
    # because `default_integer_bitwidth` is cudf only option
    assert_eq(expected, actual, exact=False)
