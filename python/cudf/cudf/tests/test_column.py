# Copyright (c) 2020, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
import cudf.utils.dtypes as dtypeutils
from cudf.core.column.column import as_column
from cudf.tests.utils import assert_eq

dtypes = sorted(
    list(
        dtypeutils.ALL_TYPES
        - {"datetime64[s]", "datetime64[ms]", "datetime64[us]"}
    )
)


@pytest.fixture(params=dtypes, ids=dtypes)
def pandas_input(request):
    data = np.random.randint(0, 1000, 100)
    return pd.Series(data, dtype=request.param)


def str_host_view(list_of_str, to_dtype):
    return np.concatenate(
        [np.frombuffer(s.encode("utf-8"), dtype=to_dtype) for s in list_of_str]
    )


@pytest.mark.parametrize("offset", [0, 1, 15])
@pytest.mark.parametrize("size", [None, 50, 10, 0])
def test_column_offset_and_size(pandas_input, offset, size):
    col = cudf.core.column.as_column(pandas_input)
    col = cudf.core.column.build_column(
        data=col.base_data,
        dtype=col.dtype,
        mask=col.base_mask,
        size=size,
        offset=offset,
        children=col.base_children,
    )

    if cudf.utils.dtypes.is_categorical_dtype(col.dtype):
        assert col.size == col.codes.size
        assert col.size == (col.codes.data.size / col.codes.dtype.itemsize)
    elif pd.api.types.is_string_dtype(col.dtype):
        assert col.size == (col.children[0].size - 1)
        assert col.size == (
            (col.children[0].data.size / col.children[0].dtype.itemsize) - 1
        )
    else:
        assert col.size == (col.data.size / col.dtype.itemsize)

    got = cudf.Series(col)

    if offset is None:
        offset = 0
    if size is None:
        size = 100
    else:
        size = size + offset

    slicer = slice(offset, size)
    expect = pandas_input.iloc[slicer].reset_index(drop=True)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[23, 68, 2, 38, 9, 83, 72, 6, 98, 30]]),
        np.array([[1, 2], [7, 6]]),
    ],
)
def test_column_series_multi_dim(data):
    with pytest.raises(ValueError):
        cudf.Series(data)

    with pytest.raises(ValueError):
        cudf.core.column.as_column(data)


@pytest.mark.parametrize("data", [["1.0", "2", -3], ["1", "0.11", 0.1]])
def test_column_series_misc_input(data):
    psr = pd.Series(data)
    sr = cudf.Series(data)

    assert_eq(psr.dtype, sr.dtype)
    assert_eq(psr.astype("str"), sr)


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_as_column_scalar_with_nan(nan_as_null):
    size = 10
    scalar = np.nan

    expected = cudf.Series([np.nan] * size, nan_as_null=nan_as_null).to_array()

    got = cudf.Series(
        as_column(scalar, length=size, nan_as_null=nan_as_null)
    ).to_array()

    np.testing.assert_equal(expected, got)


@pytest.mark.parametrize("data", [[1.1, 2.2, 3.3, 4.4], [1, 2, 3, 4]])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_column_series_cuda_array_dtype(data, dtype):
    psr = pd.Series(np.asarray(data), dtype=dtype)
    sr = cudf.Series(cp.asarray(data), dtype=dtype)

    assert_eq(psr, sr)

    psr = pd.Series(data, dtype=dtype)
    sr = cudf.Series(data, dtype=dtype)

    assert_eq(psr, sr)


def test_column_zero_length_slice():
    # see https://github.com/rapidsai/cudf/pull/4777
    from numba import cuda

    x = cudf.DataFrame({"a": [1]})
    the_column = x[1:]["a"]._column

    expect = np.array([], dtype="int8")
    got = cuda.as_cuda_array(the_column.data).copy_to_host()

    np.testing.assert_array_equal(expect, got)


def test_column_chunked_array_creation():
    pyarrow_array = pa.array([1, 2, 3] * 1000)
    chunked_array = pa.chunked_array(pyarrow_array)

    actual_column = cudf.core.column.as_column(chunked_array, dtype="float")
    expected_column = cudf.core.column.as_column(pyarrow_array, dtype="float")

    assert_eq(cudf.Series(actual_column), cudf.Series(expected_column))

    actual_column = cudf.core.column.as_column(chunked_array)
    expected_column = cudf.core.column.as_column(pyarrow_array)

    assert_eq(cudf.Series(actual_column), cudf.Series(expected_column))


@pytest.mark.parametrize(
    "data,from_dtype,to_dtype",
    [
        # equal size different kind
        (np.arange(3), "int64", "float64"),
        (np.arange(3), "float32", "int32"),
        (np.arange(1), "int64", "datetime64[ns]"),
        # size / 2^n should work for all n
        (np.arange(3), "int64", "int32"),
        (np.arange(3), "int64", "int16"),
        (np.arange(3), "int64", "int8"),
        (np.arange(3), "float64", "float32"),
        # evenly divides into bigger type
        (np.arange(8), "int8", "int64"),
        (np.arange(16), "int8", "int64"),
        (np.arange(128), "int8", "int64"),
        (np.arange(2), "float32", "int64"),
        (np.arange(8), "int8", "datetime64[ns]"),
        (np.arange(16), "int8", "datetime64[ns]"),
    ],
)
def test_column_view_valid_numeric_to_numeric(data, from_dtype, to_dtype):
    cpu_data = np.asarray(data, dtype=from_dtype)
    gpu_data = as_column(data, dtype=from_dtype)

    cpu_data_view = cpu_data.view(to_dtype)
    gpu_data_view = gpu_data.view(to_dtype)

    expect = pd.Series(cpu_data_view, dtype=cpu_data_view.dtype)
    got = cudf.Series(gpu_data_view, dtype=gpu_data_view.dtype)

    gpu_ptr = gpu_data.data.ptr
    assert gpu_ptr == got._column.data.ptr
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,from_dtype,to_dtype",
    [
        (np.arange(9), "int8", "int64"),
        (np.arange(3), "int8", "int16"),
        (np.arange(6), "int8", "float32"),
        (np.arange(2), "int64", "datetime64[ns]"),
        (np.arange(1), "int8", "datetime64[ns]"),
    ],
)
def test_column_view_invalid_numeric_to_numeric(data, from_dtype, to_dtype):
    cpu_data = np.asarray(data, dtype=from_dtype)
    gpu_data = as_column(data, dtype=from_dtype)

    try:
        cpu_data = cpu_data.view(to_dtype)
    except ValueError as error:
        if "size must be a divisor" in str(error):
            with pytest.raises(
                ValueError, match="Can not divide",
            ):
                gpu_data = gpu_data.view(to_dtype)
        else:
            raise error


@pytest.mark.parametrize(
    "data,to_dtype",
    [
        (["a", "b", "c"], "int8"),
        (["ab"], "int8"),
        (["ab"], "int16"),
        (["a", "ab", "a"], "int8"),
        (["abcd", "efgh"], "float32"),
        (["abcdefgh"], "datetime64[ns]"),
    ],
)
def test_column_view_valid_string_to_numeric(data, to_dtype):
    expect = cudf.Series(cudf.Series(data)._column.view(to_dtype))
    got = cudf.Series(str_host_view(data, to_dtype))

    assert_eq(expect, got)


def test_column_view_nulls_widths_even():

    data = [1, 2, None, 4, None]
    expect_data = [
        np.int32(val).view("float32") if val is not None else np.nan
        for val in data
    ]

    sr = cudf.Series(data, dtype="int32")
    expect = cudf.Series(expect_data, dtype="float32")
    got = cudf.Series(sr._column.view("float32"))

    assert_eq(expect, got)

    data = [None, 2.1, None, 5.3, 8.8]
    expect_data = [
        np.float64(val).view("int64") if val is not None else val
        for val in data
    ]

    sr = cudf.Series(data, dtype="float64")
    expect = cudf.Series(expect_data, dtype="int64")
    got = cudf.Series(sr._column.view("int64"))

    assert_eq(expect, got)


@pytest.mark.parametrize("slc", [slice(1, None), slice(None, 4), slice(2, 4)])
def test_column_view_numeric_slice(slc):

    data = np.array([1, 2, 3, 4, 5], dtype="int32")
    sr = cudf.Series(data)

    expect = cudf.Series(data[slc].view("int64"))
    got = cudf.Series(sr._column[slc].view("int64"))

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "slc", [slice(3, None), slice(None, 4), slice(2, 5), slice(1, 3)]
)
def test_column_view_string_slice(slc):
    data = ["a", "bcde", "cd", "efg", "h"]

    expect = cudf.Series(cudf.Series(data)._column[slc].view("int8"))
    got = cudf.Series(str_host_view(data[slc], "int8"))

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            np.array([1, 2, 3, 4, 5], dtype="uint8"),
            cudf.core.column.as_column([1, 2, 3, 4, 5], dtype="uint8"),
        ),
        (
            cp.array([1, 2, 3, 4, 5], dtype="uint8"),
            cudf.core.column.as_column([1, 2, 3, 4, 5], dtype="uint8"),
        ),
        (
            cp.array([], dtype="uint8"),
            cudf.core.column.as_column([], dtype="uint8"),
        ),
        (
            cp.array([453], dtype="uint8"),
            cudf.core.column.as_column([453], dtype="uint8"),
        ),
    ],
)
def test_as_column_buffer(data, expected):
    actual_column = cudf.core.column.as_column(
        cudf.core.Buffer(data), dtype=data.dtype
    )
    assert_eq(cudf.Series(actual_column), cudf.Series(expected))
