# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


@pytest.fixture
def test_data():
    return [[[[0, 1], [2], [5], [6, 7]], [[8], [9], [], [13, 14, 15]]]]


@pytest.fixture
def list_column():
    return [[0, 1], [2], [5], [6, 7]]


@pytest.fixture
def scalar():
    return pa.scalar(1)


@pytest.fixture
def column():
    return pa.array([3, 2, 5, 6]), pa.array([-1, 0, 0, 0], type=pa.int32())


@pytest.fixture
def bool_column():
    return pa.array([[False, True], [True], [True], [True, True]])


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


def test_contains_scalar(list_column, scalar):
    arr = pa.array(list_column)

    plc_column = plc.interop.from_arrow(arr)
    plc_scalar = plc.interop.from_arrow(scalar)
    res = plc.lists.contains(plc_column, plc_scalar)

    expect = pa.array([True, False, False, False])

    assert_column_eq(expect, res)


def test_contains_list_column(list_column, column):
    list_column1 = list_column
    list_column2, _ = column
    arr1 = pa.array(list_column1)
    arr2 = pa.array(list_column2)

    plc_column1 = plc.interop.from_arrow(arr1)
    plc_column2 = plc.interop.from_arrow(arr2)
    res = plc.lists.contains(plc_column1, plc_column2)

    expect = pa.array([False, True, True, True])

    assert_column_eq(expect, res)


@pytest.mark.parametrize(
    "list_column, expected",
    [
        (
            [[1, None], [1, 3, 4], [5, None]],
            [True, False, True],
        ),
        (
            [[1, None], None, [5]],
            [True, None, False],
        ),
    ],
)
def test_contains_nulls(list_column, expected):
    arr = pa.array(list_column)
    plc_column = plc.interop.from_arrow(arr)
    res = plc.lists.contains_nulls(plc_column)

    expect = pa.array(expected)

    assert_column_eq(expect, res)


def test_index_of_scalar(list_column, scalar):
    arr = pa.array(list_column)

    plc_column = plc.interop.from_arrow(arr)
    plc_scalar = plc.interop.from_arrow(scalar)
    res = plc.lists.index_of(plc_column, plc_scalar, True)

    expect = pa.array([1, -1, -1, -1], type=pa.int32())

    assert_column_eq(expect, res)


def test_index_of_list_column(list_column, column):
    arr1 = pa.array(list_column)
    arr2, expect = column
    plc_column1 = plc.interop.from_arrow(arr1)
    plc_column2 = plc.interop.from_arrow(arr2)
    res = plc.lists.index_of(plc_column1, plc_column2, True)

    expect = pa.array(column[1], type=pa.int32())

    assert_column_eq(expect, res)


def test_reverse(list_column):
    arr = pa.array(list_column)
    plc_column = plc.interop.from_arrow(arr)

    res = plc.lists.reverse(plc_column)

    expect = pa.array([lst[::-1] for lst in list_column])

    assert_column_eq(expect, res)


def test_segmented_gather(test_data):
    list_column1, list_column2 = test_data[0]

    plc_column1 = plc.interop.from_arrow(pa.array(list_column1))
    plc_column2 = plc.interop.from_arrow(pa.array(list_column2))

    res = plc.lists.segmented_gather(plc_column2, plc_column1)

    expect = pa.array([[8, 9], [14], [0], [0, 0]])

    assert_column_eq(expect, res)


def test_extract_list_element_scalar(list_column):
    plc_column = plc.interop.from_arrow(pa.array(list_column))

    res = plc.lists.extract_list_element(plc_column, 0)
    expect = pa.compute.list_element(list_column, 0)

    assert_column_eq(expect, res)


def test_extract_list_element_column(list_column):
    plc_column = plc.interop.from_arrow(pa.array(list_column))
    indices = plc.interop.from_arrow(pa.array([0, 1, -4, -1]))

    res = plc.lists.extract_list_element(plc_column, indices)
    expect = pa.array([0, None, None, 7])

    assert_column_eq(expect, res)


def test_count_elements(test_data):
    arr = pa.array(test_data[0][1])
    plc_column = plc.interop.from_arrow(arr)
    res = plc.lists.count_elements(plc_column)

    expect = pa.array([1, 1, 0, 3], type=pa.int32())

    assert_column_eq(expect, res)


def test_apply_boolean_mask(list_column, bool_column):
    arr = pa.array(list_column)
    plc_column = plc.interop.from_arrow(arr)
    plc_bool_column = plc.interop.from_arrow(bool_column)

    res = plc.lists.apply_boolean_mask(plc_column, plc_bool_column)

    expect = pa.array([[1], [2], [5], [6, 7]])

    assert_column_eq(expect, res)


def test_distinct():
    list_column = [[0, 1, 2, 3, 2], [3, 1, 2], None, [4, None, None, 5]]
    arr = pa.array(list_column)
    plc_column = plc.interop.from_arrow(arr)

    res = plc.lists.distinct(plc_column, True, True)

    expect = pa.array([[0, 1, 2, 3], [3, 1, 2], None, [4, None, 5]])

    assert_column_eq(expect, res)
