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
