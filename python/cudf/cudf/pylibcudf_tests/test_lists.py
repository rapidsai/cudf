# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf
from cudf._lib import pylibcudf as plc


@pytest.fixture
def test_data():
    return [[[[0, 1], [2], [5], [6, 7]], [[8], [9], [], [13, 14, 15]]]]


def test_concatenate_rows(test_data):
    arrow_tbl = pa.Table.from_arrays(test_data[0], names=["a", "b"])
    plc_tbl = plc.interop.from_arrow(arrow_tbl)

    res = plc.lists.concatenate_rows(plc_tbl)

    expect = pa.array([pair[0] + pair[1] for pair in zip(*test_data[0])])

    assert_column_eq(expect, res)


@pytest.mark.parametrize(
    "test_data, dropna, expected",
    [
        (
            [[[1, 2], [3, 4], [5]], [[6], None, [7, 8, 9]]],
            False,
            [[1, 2, 3, 4, 5], None],
        ),
        (
            [[[1, 2], [3, 4], [5, None]], [[6], [None], [7, 8, 9]]],
            True,
            [[1, 2, 3, 4, 5, None], [6, None, 7, 8, 9]],
        ),
    ],
)
def test_concatenate_list_elements(test_data, dropna, expected):
    arr = pa.array(test_data)
    plc_column = plc.interop.from_arrow(arr)

    res = plc.lists.concatenate_list_elements(plc_column, dropna)

    expect = pa.array(expected)

    assert_column_eq(expect, res)


def test_contains_scalar(test_data):
    list_column = test_data[0][0]
    arr = pa.array(list_column)

    plc_column = plc.interop.from_arrow(arr)
    value = np.int64(1)
    scalar = cudf.utils.dtypes.to_cudf_compatible_scalar(value, value.dtype)
    s = cudf._lib.scalar.DeviceScalar(scalar, value.dtype)
    res = plc.lists.contains(plc_column, s)

    expect = pa.array([True, False, False, False])

    assert_column_eq(expect, res)


def test_contains_list_column(test_data):
    list_column1 = test_data[0][0]
    list_column2 = [1, 3, 5, 1]
    arr1 = pa.array(list_column1)
    arr2 = pa.array(list_column2)

    plc_column1 = plc.interop.from_arrow(arr1)
    plc_column2 = plc.interop.from_arrow(arr2)
    res = plc.lists.contains(plc_column1, plc_column2)

    expect = pa.array([True, False, True, False])

    assert_column_eq(expect, res)


def test_contains_nulls():
    list_column = [[1, None], [1, 3, 4], [5, None]]
    arr = pa.array(list_column)
    plc_column = plc.interop.from_arrow(arr)
    res = plc.lists.contains_nulls(plc_column)

    expect = pa.array([True, False, True])

    assert_column_eq(expect, res)


def test_index_of_scalar(test_data):
    list_column = test_data[0][0]
    arr = pa.array(list_column)

    plc_column = plc.interop.from_arrow(arr)
    value = np.int64(1)
    scalar = cudf.utils.dtypes.to_cudf_compatible_scalar(value, value.dtype)
    s = cudf._lib.scalar.DeviceScalar(scalar, value.dtype)
    res = plc.lists.index_of(plc_column, s)

    expect = pa.array([1, -1, -1, -1], type=pa.int32())

    assert_column_eq(expect, res)


def test_index_of_list_column(test_data):
    list_column1 = test_data[0][0]
    list_column2 = [3, 2, 5, 6]
    arr1 = pa.array(list_column1)
    arr2 = pa.array(list_column2)

    plc_column1 = plc.interop.from_arrow(arr1)
    plc_column2 = plc.interop.from_arrow(arr2)
    res = plc.lists.index_of(plc_column1, plc_column2)

    expect = pa.array([-1, 0, 0, 0], type=pa.int32())

    assert_column_eq(expect, res)
