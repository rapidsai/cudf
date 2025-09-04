# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "lhs, rhs", [("a", "a"), ("a", "b"), (1, 1.0), (None, None), (None, "a")]
)
def test_equals_names(lhs, rhs):
    lhs = cudf.DataFrame({lhs: [1, 2]})
    rhs = cudf.DataFrame({rhs: [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


def test_equals_dtypes():
    lhs = cudf.DataFrame({"a": [1, 2.0]})
    rhs = cudf.DataFrame({"a": [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert got == expect
