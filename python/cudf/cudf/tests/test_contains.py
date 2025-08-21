# Copyright (c) 2019-2025, NVIDIA CORPORATION.


import numpy as np
import pytest

import cudf
from cudf import Series
from cudf.testing._utils import DATETIME_TYPES, NUMERIC_TYPES, TIMEDELTA_TYPES


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_lists_contains(dtype):
    dtype = cudf.dtype(dtype)
    inner_data = np.array([1, 2, 3], dtype=dtype)

    data = Series([inner_data])

    contained_scalar = inner_data.dtype.type(2)
    not_contained_scalar = inner_data.dtype.type(42)

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]


@pytest.mark.parametrize("dtype", DATETIME_TYPES + TIMEDELTA_TYPES)
def test_lists_contains_datetime(dtype):
    dtype = cudf.dtype(dtype)
    inner_data = np.array([1, 2, 3])

    unit, _ = np.datetime_data(dtype)

    data = Series([inner_data])

    contained_scalar = inner_data.dtype.type(2)
    not_contained_scalar = inner_data.dtype.type(42)

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]


def test_lists_contains_bool():
    data = Series([[True, True, True]])

    contained_scalar = True
    not_contained_scalar = False

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]
