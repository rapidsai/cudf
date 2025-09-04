# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf


@pytest.mark.parametrize("data", [range(-3, 3), range(1, 3), range(0)])
def test_rangeindex_all(data):
    result = cudf.RangeIndex(data).all()
    expected = cudf.Index(list(data)).all()
    assert result == expected
