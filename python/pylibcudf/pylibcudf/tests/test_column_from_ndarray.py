# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc

NUMPY_DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.bool_,
]


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def np_1darray(request):
    dtype = request.param
    return np.array([0, 1, 2, 3], dtype=dtype)


@pytest.fixture
def cp_1darray(np_1darray):
    return cp.asarray(np_1darray)


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def np_2darray(request):
    dtype = request.param
    return np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype)


@pytest.fixture
def cp_2darray(np_2darray):
    return cp.asarray(np_2darray)


def test_from_ndarray_cupy_1d(cp_1darray, np_1darray):
    expect = pa.array(
        np_1darray, type=pa.from_numpy_dtype(np_1darray.dtype.type)
    )
    got = plc.Column.from_ndarray(cp_1darray)
    assert_column_eq(expect, got)


def test_from_ndarray_cupy_2d(cp_2darray, np_2darray):
    dtype = pa.from_numpy_dtype(np_2darray.dtype.type)
    expect = pa.array(np_2darray.tolist(), type=pa.list_(dtype))
    got = plc.Column.from_ndarray(cp_2darray)
    assert_column_eq(expect, got)


def test_from_ndarray_numpy_1d(np_1darray):
    with pytest.raises(
        NotImplementedError, match="Converting to a .* not yet implemented."
    ):
        plc.Column.from_ndarray(np_1darray)


def test_from_ndarray_numpy_2d(np_2darray):
    with pytest.raises(
        NotImplementedError, match="Converting to a .* not yet implemented."
    ):
        plc.Column.from_ndarray(np_2darray)


@pytest.mark.parametrize("obj", [None, "str"])
def test_from_ndarray_invalid_obj(obj):
    with pytest.raises(
        TypeError, match="Cannot convert a .* to a pylibcudf Column"
    ):
        plc.Column.from_ndarray(obj)
