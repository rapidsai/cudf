# Copyright (c) 2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc

np = pytest.importorskip("numpy")
cp = pytest.importorskip("cupy")

CUPY_DTYPES = [
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

NUMPY_DTYPES = [
    *CUPY_DTYPES,
    *[
        np.dtype("datetime64[s]"),
        np.dtype("datetime64[ms]"),
        np.dtype("datetime64[us]"),
        np.dtype("datetime64[ns]"),
        np.dtype("timedelta64[s]"),
        np.dtype("timedelta64[ms]"),
        np.dtype("timedelta64[us]"),
        np.dtype("timedelta64[ns]"),
    ],
]


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def np_1darray(request):
    dtype = request.param
    return np.array([0, 1, 2, 3], dtype=dtype)


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def np_2darray(request):
    dtype = request.param
    return np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype)


@pytest.fixture(params=CUPY_DTYPES, ids=repr)
def cp_1darray(request):
    dtype = request.param
    np_data = np.array([0, 1, 2, 3], dtype=dtype)
    return cp.asarray(np_data), np_data


@pytest.fixture(params=CUPY_DTYPES, ids=repr)
def cp_2darray(request):
    dtype = request.param
    np_data = np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype)
    return cp.asarray(np_data), np_data


def test_from_ndarray_cupy_1d(cp_1darray):
    cp_arr, np_arr = cp_1darray
    expect = pa.array(np_arr, type=pa.from_numpy_dtype(np_arr.dtype))
    got = plc.Column.from_array(cp_arr)
    assert_column_eq(expect, got)


def test_from_ndarray_cupy_2d(cp_2darray):
    cp_arr, np_arr = cp_2darray
    dtype = pa.from_numpy_dtype(np_arr.dtype)
    expect = pa.array(np_arr.tolist(), type=pa.list_(dtype))
    got = plc.Column.from_array(cp_arr)
    assert_column_eq(expect, got)


def test_from_ndarray_numpy_1d(np_1darray):
    expect = pa.array(np_1darray, type=pa.from_numpy_dtype(np_1darray.dtype))
    got = plc.Column.from_array(np_1darray)
    assert_column_eq(expect, got)


def test_from_ndarray_numpy_2d(np_2darray):
    dtype = pa.from_numpy_dtype(np_2darray.dtype)
    expect = pa.array(np_2darray.tolist(), type=pa.list_(dtype))
    got = plc.Column.from_array(np_2darray)
    assert_column_eq(expect, got)


def test_non_c_contiguous_raises(cp_2darray):
    with pytest.raises(
        ValueError,
        match="Data must be C-contiguous",
    ):
        plc.Column.from_array(cp.asfortranarray(cp_2darray[0]))


def test_row_limit_exceed_raises():
    with pytest.raises(
        ValueError,
        match="Number of rows exceeds size_type limit",
    ):
        plc.Column.from_array(cp.zeros((2**31, 1)))


@pytest.mark.parametrize("obj", [None, "str"])
def test_from_ndarray_invalid_obj(obj):
    with pytest.raises(
        TypeError,
        match="Cannot convert object of type .* to a pylibcudf Column",
    ):
        plc.Column.from_array(obj)


def test_array_interface_with_data_none():
    class ArrayInterfaceWithNone:
        @property
        def __array_interface__(self):
            return {
                "shape": (4,),
                "typestr": "<i4",
                "data": None,
                "version": 3,
            }

    with pytest.raises(
        ValueError,
        match="Expected a data field .* the array interface.",
    ):
        plc.Column.from_array(ArrayInterfaceWithNone())
