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


class CuPyArrayWithMask(cp.ndarray):
    def __new__(cls, input_array, mask_array):
        obj = cp.asarray(input_array).view(cls)
        obj._mask = mask_array
        return obj

    def ravel(self, order="C"):
        flat_data = super().ravel(order)
        return CuPyArrayWithMask(flat_data, self._mask)

    @property
    def __cuda_array_interface__(self):
        iface = super().__cuda_array_interface__.copy()
        iface["mask"] = self._mask
        return iface


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def cp_1darray_with_mask(request):
    dtype = request.param
    data = cp.array([0, 1, 2, 3], dtype=dtype)
    mask = cp.array([1, 0, 1, 1], dtype=cp.bool_)
    return CuPyArrayWithMask(data, mask)


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def cp_2darray_with_mask(request):
    dtype = request.param
    data = cp.array([[0, 1, 2], [3, 4, 5]], dtype=dtype)
    mask = cp.array([[1, 0, 1], [0, 1, 1]], dtype=cp.bool_)
    return CuPyArrayWithMask(data, mask)


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


def test_from_ndarray_cupy_1d_with_mask(cp_1darray_with_mask):
    np_data = cp.asnumpy(cp_1darray_with_mask)
    np_mask = cp.asnumpy(cp_1darray_with_mask._mask).astype(bool)
    expect = pa.array(
        np_data, type=pa.from_numpy_dtype(np_data.dtype), mask=~np_mask
    )
    got = plc.Column.from_ndarray(cp_1darray_with_mask)
    assert_column_eq(expect, got)


def test_from_ndarray_cupy_2d_with_mask(cp_2darray_with_mask):
    got = plc.interop.to_arrow(plc.Column.from_ndarray(cp_2darray_with_mask))

    # pyarrow doesn't support creating a
    # 2d array with a boolean mask. So instead we
    # verify the correctness of resulting pylibcudf
    # Column's null count.
    assert pa.compute.count(got).as_py() == 2
