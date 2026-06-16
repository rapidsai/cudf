# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


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
def search_key_column():
    return pa.array([3, 2, 5, 6]), pa.array([-1, 0, 0, 0], type=pa.int32())


@pytest.fixture
def bool_column():
    return pa.array([[False, True], [True], [True], [True, True]])


@pytest.fixture
def set_lists_column():
    lhs = [[np.nan, np.nan, 2, 1, 2], [1, 2, 3], None, [4, None, 5]]
    rhs = [[np.nan, 1, 2, 3], [4, 5], [None, 7, 8], [None, None]]
    return lhs, rhs


@pytest.fixture
def lists_column():
    return [[4, 2, 3, 1], [1, 2, None, 4], [-10, 10, 10, 0]]


def test_concatenate_rows(test_data):
    plc_tbl = plc.Table.from_arrow(
        pa.Table.from_arrays(test_data[0], names=["a", "b"])
    )

    got = plc.lists.concatenate_rows(plc_tbl)

    expect = pa.array(
        [pair[0] + pair[1] for pair in zip(*test_data[0], strict=True)]
    )

    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "test_data, dropna, expected",
    [
        (
            [[[1, 2], [3, 4], [5]], [[6], None, [7, 8, 9]]],
            plc.lists.ConcatenateNullPolicy.NULLIFY_OUTPUT_ROW,
            [[1, 2, 3, 4, 5], None],
        ),
        (
            [[[1, 2], [3, 4], [5, None]], [[6], [None], [7, 8, 9]]],
            plc.lists.ConcatenateNullPolicy.IGNORE,
            [[1, 2, 3, 4, 5, None], [6, None, 7, 8, 9]],
        ),
    ],
)
def test_concatenate_list_elements(test_data, dropna, expected):
    got = plc.lists.concatenate_list_elements(
        plc.Column.from_arrow(pa.array(test_data)), dropna
    )

    expect = pa.array(expected)

    assert_column_eq(expect, got)


def test_contains_scalar(list_column, scalar):
    plc_column = plc.Column.from_arrow(pa.array(list_column))
    plc_scalar = plc.Scalar.from_arrow(scalar)
    got = plc.lists.contains(plc_column, plc_scalar)

    expect = pa.array([True, False, False, False])

    assert_column_eq(expect, got)


def test_contains_list_column(list_column, search_key_column):
    plc_column1 = plc.Column.from_arrow(pa.array(list_column))
    plc_column2 = plc.Column.from_arrow(pa.array(search_key_column[0]))
    got = plc.lists.contains(plc_column1, plc_column2)

    expect = pa.array([False, True, True, True])

    assert_column_eq(expect, got)


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
    plc_column = plc.Column.from_arrow(pa.array(list_column))
    got = plc.lists.contains_nulls(plc_column)

    expect = pa.array(expected)

    assert_column_eq(expect, got)


def test_index_of_scalar(list_column, scalar):
    arr = pa.array(list_column)

    plc_column = plc.Column.from_arrow(arr)
    plc_scalar = plc.Scalar.from_arrow(scalar)
    got = plc.lists.index_of(
        plc_column, plc_scalar, plc.lists.DuplicateFindOption.FIND_FIRST
    )

    expect = pa.array([1, -1, -1, -1], type=pa.int32())

    assert_column_eq(expect, got)


def test_index_of_list_column(list_column, search_key_column):
    plc_column1 = plc.Column.from_arrow(pa.array(list_column))
    plc_column2 = plc.Column.from_arrow(search_key_column[0])
    got = plc.lists.index_of(
        plc_column1, plc_column2, plc.lists.DuplicateFindOption.FIND_FIRST
    )

    expect = pa.array(search_key_column[1], type=pa.int32())

    assert_column_eq(expect, got)


def test_reverse(list_column):
    plc_column = plc.Column.from_arrow(pa.array(list_column))

    got = plc.lists.reverse(plc_column)

    expect = pa.array([lst[::-1] for lst in list_column])

    assert_column_eq(expect, got)


@pytest.fixture
def segmented_gather_input_data() -> list[list[str | None]]:
    return [["a", "b"], ["c"], [], ["d", None, "e"]]


@pytest.mark.parametrize(
    "bounds_policy",
    [
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        plc.copying.OutOfBoundsPolicy.NULLIFY,
    ],
    ids=["DONT_CHECK", "NULLIFY"],
)
def test_segmented_gather_in_bounds(
    segmented_gather_input_data: list[list[str | None]],
    bounds_policy: plc.copying.OutOfBoundsPolicy,
) -> None:
    input_column = plc.Column.from_arrow(pa.array(segmented_gather_input_data))
    # these are all in-bounds for input_column.
    gather_map_list = plc.Column.from_arrow(
        pa.array([[0, 0], [0], [], [1, 2]])
    )
    got = plc.lists.segmented_gather(
        input_column, gather_map_list, bounds_policy=bounds_policy
    )

    expect = pa.array([["a", "a"], ["c"], [], [None, "e"]])

    assert_column_eq(expect, got)


def test_segmented_gather_out_of_bounds(
    segmented_gather_input_data: list[list[str | None]],
) -> None:
    input_column = plc.Column.from_arrow(pa.array(segmented_gather_input_data))
    gather_map_list = plc.Column.from_arrow(
        pa.array([[0, 1], [2], [], [-5, 1, 2]])
    )
    got = plc.lists.segmented_gather(
        input_column, gather_map_list, plc.copying.OutOfBoundsPolicy.NULLIFY
    )

    expect = pa.array([["a", "b"], [None], [], [None, None, "e"]])

    assert_column_eq(expect, got)


def test_extract_list_element_column(list_column):
    plc_column = plc.Column.from_arrow(pa.array(list_column))
    indices = plc.Column.from_arrow(pa.array([0, 1, -4, -1]))

    got = plc.lists.extract_list_element(plc_column, indices)
    expect = pa.array([0, None, None, 7])

    assert_column_eq(expect, got)


def test_count_elements(test_data):
    arr = pa.array(test_data[0][1])
    plc_column = plc.Column.from_arrow(arr)
    got = plc.lists.count_elements(plc_column)

    expect = pa.array([1, 1, 0, 3], type=pa.int32())

    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "steps, expect",
    [
        (None, pa.array([[], [1, 2], [2, 3], [3], [4, 5, 6]])),
        (
            pa.array([2, 1, 1, 1, -3]),
            pa.array([[], [1, 2], [2, 3], [3], [4, 1, -2]]),
        ),
    ],
)
def test_sequences_parametrized(steps, expect):
    starts = plc.Column.from_arrow(pa.array([0, 1, 2, 3, 4]))
    sizes = plc.Column.from_arrow(pa.array([0, 2, 2, 1, 3]))
    steps = plc.Column.from_arrow(steps) if steps is not None else None

    got = (
        plc.lists.sequences(starts, sizes)
        if steps is None
        else plc.lists.sequences(starts, sizes, steps)
    )

    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "order, na_position, expect",
    [
        (
            plc.types.Order.ASCENDING,
            plc.types.NullOrder.BEFORE,
            [[1, 2, 3, 4], [None, 1, 2, 4], [-10, 0, 10, 10]],
        ),
        (
            plc.types.Order.ASCENDING,
            plc.types.NullOrder.AFTER,
            [[1, 2, 3, 4], [1, 2, 4, None], [-10, 0, 10, 10]],
        ),
        (
            plc.types.Order.DESCENDING,
            plc.types.NullOrder.BEFORE,
            [[4, 3, 2, 1], [4, 2, 1, None], [10, 10, 0, -10]],
        ),
        (
            plc.types.Order.DESCENDING,
            plc.types.NullOrder.AFTER,
            [[4, 3, 2, 1], [None, 4, 2, 1], [10, 10, 0, -10]],
        ),
    ],
)
@pytest.mark.parametrize("stable", [False, True])
def test_sort_lists_param(lists_column, order, na_position, expect, stable):
    plc_column = plc.Column.from_arrow(pa.array(lists_column))
    got = plc.lists.sort_lists(plc_column, order, na_position, stable)
    expect = pa.array(expect)

    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "set_operation,nans_equal,nulls_equal,expected",
    [
        (
            plc.lists.difference_distinct,
            plc.types.NanEquality.ALL_EQUAL,
            plc.types.NullEquality.EQUAL,
            [[], [1, 2, 3], None, [4, 5]],
        ),
        (
            plc.lists.difference_distinct,
            plc.types.NanEquality.UNEQUAL,
            plc.types.NullEquality.EQUAL,
            [[], [1, 2, 3], None, [4, None, 5]],
        ),
        (
            plc.lists.have_overlap,
            plc.types.NanEquality.ALL_EQUAL,
            plc.types.NullEquality.EQUAL,
            [True, False, None, True],
        ),
        (
            plc.lists.have_overlap,
            plc.types.NanEquality.UNEQUAL,
            plc.types.NullEquality.UNEQUAL,
            [True, False, None, False],
        ),
        (
            plc.lists.intersect_distinct,
            plc.types.NanEquality.ALL_EQUAL,
            plc.types.NullEquality.EQUAL,
            [[np.nan, 1, 2], [], None, [None]],
        ),
        (
            plc.lists.intersect_distinct,
            plc.types.NanEquality.ALL_EQUAL,
            plc.types.NullEquality.UNEQUAL,
            [[1, 2], [], None, [None]],
        ),
        (
            plc.lists.union_distinct,
            plc.types.NanEquality.UNEQUAL,
            plc.types.NullEquality.EQUAL,
            [
                [np.nan, 2, 1, 3],
                [1, 2, 3, 4, 5],
                None,
                [4, None, 5, None, None],
            ],
        ),
        (
            plc.lists.union_distinct,
            plc.types.NanEquality.UNEQUAL,
            plc.types.NullEquality.UNEQUAL,
            [
                [np.nan, np.nan, 2, 1, np.nan, 3],
                [1, 2, 3, 4, 5],
                None,
                [4, None, 5, None, None],
            ],
        ),
    ],
)
def test_set_operations(
    set_lists_column, set_operation, nans_equal, nulls_equal, expected
):
    lhs, rhs = set_lists_column

    got = set_operation(
        plc.Column.from_arrow(pa.array(lhs)),
        plc.Column.from_arrow(pa.array(rhs)),
        nans_equal,
        nulls_equal,
    )

    if set_operation != plc.lists.have_overlap:
        expect = pa.array(expected, type=pa.list_(pa.float64()))
    else:
        expect = pa.array(expected)
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "nans_equal,nulls_equal,expected",
    [
        (
            plc.types.NanEquality.ALL_EQUAL,
            plc.types.NullEquality.EQUAL,
            [[np.nan, 0, 1, 2, 3], [3, 1, 2], None, [4, None, 5]],
        ),
        (
            plc.types.NanEquality.UNEQUAL,
            plc.types.NullEquality.EQUAL,
            [[np.nan, 0, 1, 2, 3], [3, 1, 2], None, [4, None, None, 5]],
        ),
        (
            plc.types.NanEquality.ALL_EQUAL,
            plc.types.NullEquality.UNEQUAL,
            [[np.nan, np.nan, 0, 1, 2, 3], [3, 1, 2], None, [4, None, 5]],
        ),
        (
            plc.types.NanEquality.UNEQUAL,
            plc.types.NullEquality.UNEQUAL,
            [
                [np.nan, np.nan, 0, 1, 2, 3],
                [3, 1, 2],
                None,
                [4, None, None, 5],
            ],
        ),
    ],
)
def test_distinct(list_column, nans_equal, nulls_equal, expected):
    plc_column = plc.Column.from_arrow(
        pa.array(
            [
                [np.nan, np.nan, 0, 1, 2, 3, 2],
                [3, 1, 2],
                None,
                [4, None, None, 5],
            ]
        )
    )

    got = plc.lists.distinct(plc_column, nans_equal, nulls_equal)

    expect = pa.array(expected)

    assert_column_eq(expect, got)
