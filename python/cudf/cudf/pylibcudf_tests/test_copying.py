# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import (
    assert_array_eq,
    assert_table_eq,
    column_from_arrow,
    column_to_arrow,
    scalar_to_arrow,
    table_to_arrow,
)

from cudf._lib import pylibcudf as plc


@pytest.fixture(scope="module")
def input_column():
    return column_from_arrow(pa.array([1, 2, 3]))


@pytest.fixture(scope="module")
def target_column():
    return column_from_arrow(pa.array([4, 5, 6, 7, 8, 9]))


@pytest.fixture
def mutable_target_column(target_column):
    return target_column.copy()


@pytest.fixture(scope="module")
def source_table(input_column):
    return plc.Table([input_column] * 3)


@pytest.fixture(scope="module")
def target_table(target_column):
    return plc.Table([target_column] * 3)


@pytest.fixture(scope="module")
def source_scalar():
    return plc.Scalar.from_arrow(pa.scalar(1))


def test_gather(target_table, input_column):
    result = plc.copying.gather(
        target_table,
        input_column,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    )
    pa_table = table_to_arrow(target_table)
    expected = pa_table.take(column_to_arrow(input_column))
    assert_table_eq(result, expected)


def test_scatter_table(source_table, input_column, target_table):
    result = plc.copying.scatter(
        source_table,
        input_column,
        target_table,
    )
    pa_source_table = table_to_arrow(source_table)
    pa_input_column = column_to_arrow(input_column)
    pa_target_table = table_to_arrow(target_table)

    # TODO: Is there a cleaner vectorized way to do this in pyarrow?
    output_rows = []
    for i in range(target_table.num_rows()):
        if (source_row := pa.compute.index_in(i, pa_input_column)).is_valid:
            output_rows.append(pa_source_table.take([source_row]))
        else:
            output_rows.append(pa_target_table.take([i]))

    expected = pa.concat_tables(output_rows)
    assert_table_eq(result, expected)


def test_scatter_scalars(source_scalar, input_column, target_table):
    result = plc.copying.scatter(
        [source_scalar] * target_table.num_columns(),
        input_column,
        target_table,
    )
    # TODO: There's no reason to require column metadata in this API. We can construct a
    # placeholder internally and discard it after the conversion if necessary.
    host_scalar = scalar_to_arrow(source_scalar).as_py()
    pa_input_column = column_to_arrow(input_column)
    pa_target_table = table_to_arrow(target_table)

    # TODO: Is there a cleaner vectorized way to do this in pyarrow?
    arrays = []
    for i in range(target_table.num_columns()):
        values = []
        for j in range(target_table.num_rows()):
            if pa.compute.is_in(j, pa_input_column).as_py():
                values.append(host_scalar)
            else:
                values.append(pa_target_table.column(i)[j].as_py())
        arrays.append(pa.array(values))

    expected = pa.table(arrays, [""] * target_table.num_columns())
    assert_table_eq(result, expected)


def test_empty_like(input_column):
    result = plc.copying.empty_like(input_column)
    assert input_column.type() == result.type()


def test_allocate_like(input_column):
    result = plc.copying.allocate_like(
        input_column, plc.copying.MaskAllocationPolicy.RETAIN
    )
    assert input_column.type() == result.type()
    assert input_column.size() == result.size()


def test_copy_range(input_column, target_column):
    result = plc.copying.copy_range(
        input_column,
        target_column,
        0,
        input_column.size(),
        0,
    )
    expected = pa.array([1, 2, 3, 7, 8, 9])
    assert_array_eq(result, expected)


def test_copy_range_in_place(input_column, mutable_target_column):
    plc.copying.copy_range_in_place(
        input_column,
        mutable_target_column,
        0,
        input_column.size(),
        0,
    )
    expected = pa.array([1, 2, 3, 7, 8, 9])
    assert_array_eq(mutable_target_column, expected)


def test_shift(target_column, source_scalar):
    result = plc.copying.shift(target_column, 2, source_scalar)
    expected = pa.array([1, 1, 4, 5, 6, 7])
    assert_array_eq(result, expected)


def test_split_column(target_column):
    upper_bounds = [1, 3, 5]
    lower_bounds = [0] + upper_bounds[:-1]
    result = plc.copying.split(target_column, upper_bounds)
    pa_target_column = column_to_arrow(target_column)
    for lb, ub, split in zip(lower_bounds, upper_bounds, result):
        assert_array_eq(split, pa_target_column[lb:ub])


def test_split_table(target_table):
    upper_bounds = [1, 3, 5]
    lower_bounds = [0] + upper_bounds[:-1]
    result = plc.copying.split(target_table, upper_bounds)
    pa_target_table = table_to_arrow(target_table)
    for lb, ub, split in zip(lower_bounds, upper_bounds, result):
        assert_table_eq(split, pa_target_table[lb:ub])


def test_slice_column(target_column):
    bounds = list(range(6))
    upper_bounds = bounds[1::2]
    lower_bounds = bounds[::2]
    result = plc.copying.slice(target_column, bounds)
    pa_target_column = column_to_arrow(target_column)
    for lb, ub, slice_ in zip(lower_bounds, upper_bounds, result):
        assert_array_eq(slice_, pa_target_column[lb:ub])


def test_slice_table(target_table):
    bounds = list(range(6))
    upper_bounds = bounds[1::2]
    lower_bounds = bounds[::2]
    result = plc.copying.slice(target_table, bounds)
    pa_target_table = table_to_arrow(target_table)
    for lb, ub, slice_ in zip(lower_bounds, upper_bounds, result):
        assert_table_eq(slice_, pa_target_table[lb:ub])


@pytest.mark.parametrize("array_left", [True, False])
def test_copy_if_else_column_scalar(target_column, source_scalar, array_left):
    pa_mask = pa.array([True, False] * (target_column.size() // 2))
    mask = column_from_arrow(pa_mask)
    args = (
        (target_column, source_scalar)
        if array_left
        else (source_scalar, target_column)
    )
    result = plc.copying.copy_if_else(
        *args,
        mask,
    )

    pa_scalar = source_scalar.to_arrow(plc.interop.ColumnMetadata(""))
    pa_target_column = column_to_arrow(target_column)
    pa_args = (
        (pa_target_column, pa_scalar)
        if array_left
        else (pa_scalar, pa_target_column)
    )
    expected = pa.compute.if_else(
        pa_mask,
        *pa_args,
    )
    assert_array_eq(result, expected)


def test_boolean_mask_scatter_from_table(source_table, target_table):
    py_mask = [False] * target_table.num_rows()
    py_mask[2:5] = [True, True, True]
    pa_mask = pa.array(py_mask)

    mask = column_from_arrow(pa_mask)
    result = plc.copying.boolean_mask_scatter(
        source_table,
        target_table,
        mask,
    )

    # TODO: Is there a cleaner vectorized way to do this in pyarrow?
    pa_source_table = table_to_arrow(source_table)
    pa_target_table = table_to_arrow(target_table)
    output_rows = []
    source_index = 0
    for target_index, mask_val in enumerate(pa_mask.to_pylist()):
        if mask_val:
            output_rows.append(pa_source_table.take([source_index]))
            source_index += 1
        else:
            output_rows.append(pa_target_table.take([target_index]))

    expected = pa.concat_tables(output_rows)
    assert_table_eq(result, expected)


def test_boolean_mask_scatter_from_scalars(source_scalar, target_table):
    py_mask = [False] * target_table.num_rows()
    py_mask[2:5] = [True, True, True]
    pa_mask = pa.array(py_mask)

    mask = column_from_arrow(pa_mask)
    result = plc.copying.boolean_mask_scatter(
        [source_scalar] * 3,
        target_table,
        mask,
    )

    # TODO: Is there a cleaner vectorized way to do this in pyarrow?
    host_scalar = scalar_to_arrow(source_scalar).as_py()
    pa_target_table = table_to_arrow(target_table)
    arrays = []
    for i in range(target_table.num_columns()):
        values = []
        for target_index, mask_val in enumerate(pa_mask.to_pylist()):
            if mask_val:
                values.append(host_scalar)
            else:
                values.append(pa_target_table.column(i)[target_index].as_py())
        arrays.append(pa.array(values))

    expected = pa.table(arrays, [""] * target_table.num_columns())
    assert_table_eq(result, expected)


def test_get_element(input_column):
    index = 1
    result = plc.copying.get_element(input_column, index)
    pa_input_column = column_to_arrow(input_column)
    assert scalar_to_arrow(result).as_py() == pa_input_column[index].as_py()
