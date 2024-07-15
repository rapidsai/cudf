# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


@pytest.fixture
def test_data():
    return [[[[0, 1], [2], [5], [6, 7]], [[8], [9], [], [13, 14, 15]]]]


@pytest.fixture
def scalar():
    return pa.scalar(1)


@pytest.fixture
def column():
    return pa.array([3, 2, 5, 6]), pa.array([-1, 0, 0, 0], type=pa.int32())


@pytest.fixture
def set_lists_column():
    lhs = [[2, 1, 2], [1, 2, 3], None, [4, None, 5]]
    rhs = [[1, 2, 3], [4, 5], [None, 7, 8], [None, None]]
    return lhs, rhs


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


def test_contains_scalar(test_data, scalar):
    list_column = test_data[0][0]
    arr = pa.array(list_column)

    plc_column = plc.interop.from_arrow(arr)
    plc_scalar = plc.interop.from_arrow(scalar)
    res = plc.lists.contains(plc_column, plc_scalar)

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


def test_index_of_scalar(test_data, scalar):
    list_column = test_data[0][0]
    arr = pa.array(list_column)

    plc_column = plc.interop.from_arrow(arr)
    plc_scalar = plc.interop.from_arrow(scalar)
    res = plc.lists.index_of(plc_column, plc_scalar, True)

    expect = pa.array([1, -1, -1, -1], type=pa.int32())

    assert_column_eq(expect, res)


def test_index_of_list_column(test_data, column):
    list_column = test_data[0][0]
    arr1 = pa.array(list_column)
    arr2, expect = column
    plc_column1 = plc.interop.from_arrow(arr1)
    plc_column2 = plc.interop.from_arrow(arr2)
    res = plc.lists.index_of(plc_column1, plc_column2, True)

    expect = pa.array(column[1], type=pa.int32())

    assert_column_eq(expect, res)


def test_reverse(test_data):
    list_column = test_data[0][0]
    arr = pa.array(list_column)
    plc_column = plc.interop.from_arrow(arr)

    res = plc.lists.reverse(plc_column)

    expect = pa.array([lst[::-1] for lst in list_column])

    assert_column_eq(expect, res)


def test_segmented_gather(test_data):
    list_column1 = test_data[0][0]
    list_column2 = test_data[0][1]

    plc_column1 = plc.interop.from_arrow(pa.array(list_column1))
    plc_column2 = plc.interop.from_arrow(pa.array(list_column2))

    res = plc.lists.segmented_gather(plc_column2, plc_column1)

    expect = pa.array([[8, 9], [14], [0], [0, 0]])

    assert_column_eq(expect, res)


def test_extract_list_element_scalar(test_data):
    arr = pa.array(test_data[0][0])
    plc_column = plc.interop.from_arrow(arr)

    res = plc.lists.extract_list_element(plc_column, 0)
    expect = pa.compute.list_element(test_data[0][0], 0)

    assert_column_eq(expect, res)


def test_extract_list_element_column(test_data):
    arr = pa.array(test_data[0][0])
    plc_column = plc.interop.from_arrow(arr)
    indices = plc.interop.from_arrow(pa.array([0, 1, -4, -1]))

    res = plc.lists.extract_list_element(plc_column, indices)
    expect = pa.array([0, None, None, 7])

    assert_column_eq(expect, res)


@pytest.mark.parametrize(
    "set_operation,nans_equal,nulls_equal,expected",
    [
        (
            plc.lists.difference_distinct,
            True,
            True,
            [[], [1, 2, 3], None, [4, 5]],
        ),
        (
            plc.lists.have_overlap,
            True,
            True,
            [True, False, None, True],
        ),
        (
            plc.lists.intersect_distinct,
            True,
            True,
            [[1, 2], [], None, [None]],
        ),
        (
            plc.lists.union_distinct,
            False,
            True,
            [[2, 1, 3], [1, 2, 3, 4, 5], None, [4, None, 5, None, None]],
        ),
    ],
)
def test_set_operations(
    set_lists_column, set_operation, nans_equal, nulls_equal, expected
):
    lhs, rhs = set_lists_column

    res = set_operation(
        plc.interop.from_arrow(pa.array(lhs)),
        plc.interop.from_arrow(pa.array(rhs)),
        nans_equal,
        nulls_equal,
    )
    expect = pa.array(expected)
    assert_column_eq(expect, res)
