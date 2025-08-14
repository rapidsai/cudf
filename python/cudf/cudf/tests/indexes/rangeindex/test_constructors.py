# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf


def test_rangeindex_arg_validation():
    with pytest.raises(TypeError):
        cudf.RangeIndex("1")

    with pytest.raises(TypeError):
        cudf.RangeIndex(1, "2")

    with pytest.raises(TypeError):
        cudf.RangeIndex(1, 3, "1")

    with pytest.raises(ValueError):
        cudf.RangeIndex(1, dtype="float64")

    with pytest.raises(ValueError):
        cudf.RangeIndex(1, dtype="uint64")


def test_rangeindex_name_not_hashable():
    with pytest.raises(ValueError):
        cudf.RangeIndex(range(2), name=["foo"])

    with pytest.raises(ValueError):
        cudf.RangeIndex(range(2)).copy(name=["foo"])
