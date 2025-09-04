# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "op, expected, expected_kind",
    [
        (lambda idx: 2**idx, [2, 4, 8, 16], "int"),
        (lambda idx: idx**2, [1, 4, 9, 16], "int"),
        (lambda idx: idx / 2, [0.5, 1, 1.5, 2], "float"),
        (lambda idx: 2 / idx, [2, 1, 2 / 3, 0.5], "float"),
        (lambda idx: idx % 3, [1, 2, 0, 1], "int"),
        (lambda idx: 3 % idx, [0, 1, 0, 3], "int"),
    ],
)
def test_rangeindex_binops_user_option(
    op, expected, expected_kind, default_integer_bitwidth
):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for binary operation.
    idx = cudf.RangeIndex(1, 5)
    actual = op(idx)
    expected = cudf.Index(
        expected, dtype=f"{expected_kind}{default_integer_bitwidth}"
    )
    assert_eq(
        expected,
        actual,
    )
