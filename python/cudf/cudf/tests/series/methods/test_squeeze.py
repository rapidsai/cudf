# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("axis", [None, 0, "index"])
@pytest.mark.parametrize("data", [[1, 2], [1]])
def test_squeeze(axis, data):
    ser = cudf.Series(data)
    result = ser.squeeze(axis=axis)
    expected = ser.to_pandas().squeeze(axis=axis)
    assert_eq(result, expected)


@pytest.mark.parametrize("axis", [1, "columns"])
def test_squeeze_invalid_axis(axis):
    with pytest.raises(ValueError):
        cudf.Series([1]).squeeze(axis=axis)
