# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "indices",
    [slice(0, 3), slice(1, 4), slice(None, None, 2), slice(1, None, 2)],
    ids=[":3", "1:4", "0::2", "1::2"],
)
@pytest.mark.parametrize(
    "values",
    [[None, {}, {}, None], [{}, {}, {}, {}]],
    ids=["nulls", "no_nulls"],
)
def test_struct_empty_children_slice(indices, values):
    s = cudf.Series(values)
    actual = s.iloc[indices]
    expect = cudf.Series(values[indices], index=range(len(values))[indices])
    assert_eq(actual, expect)
