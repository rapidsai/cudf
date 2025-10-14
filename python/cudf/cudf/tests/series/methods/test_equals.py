# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "lhs, rhs", [("a", "a"), ("a", "b"), (1, 1.0), (None, None), (None, "a")]
)
def test_equals_names(lhs, rhs):
    lhs = cudf.Series([1, 2], name=lhs)
    rhs = cudf.Series([1, 2], name=rhs)

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)
