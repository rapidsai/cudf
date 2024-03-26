# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import (
    DEFAULT_STRUCT_TESTING_TYPE,
    assert_column_eq,
    assert_table_eq,
    cudf_raises,
    is_fixed_width,
    is_floating,
    is_integer,
    is_string,
    metadata_from_arrow_array,
)

from cudf._lib import pylibcudf as plc


# TODO: Test nullable data
@pytest.fixture(scope="module")
def pa_input_column(pa_type):
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        return pa.array([1, 2, 3], type=pa_type)
    elif pa.types.is_string(pa_type):
        return pa.array(["a", "b", "c"], type=pa_type)
    elif pa.types.is_boolean(pa_type):
        return pa.array([True, True, False], type=pa_type)
    elif pa.types.is_list(pa_type):
        # TODO: Add heterogenous sizes
        return pa.array([[1], [2], [3]], type=pa_type)
    elif pa.types.is_struct(pa_type):
        return pa.array([{"v": 1}, {"v": 2}, {"v": 3}], type=pa_type)
    raise ValueError("Unsupported type")


@pytest.fixture(scope="module")
def input_column(pa_input_column):
    return plc.interop.from_arrow(pa_input_column)


@pytest.fixture(scope="module")
def pa_index_column():
    # Index column for testing gather/scatter, always integral.
    return pa.array([1, 2, 3])


@pytest.fixture(scope="module")
def index_column(pa_index_column):
    return plc.interop.from_arrow(pa_index_column)


@pytest.fixture(scope="module")
def pa_target_column(pa_type):
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        return pa.array([4, 5, 6, 7, 8, 9], type=pa_type)
    elif pa.types.is_string(pa_type):
        return pa.array(["d", "e", "f", "g", "h", "i"], type=pa_type)
    elif pa.types.is_boolean(pa_type):
        return pa.array([False, True, True, False, True, False], type=pa_type)
    elif pa.types.is_list(pa_type):
        # TODO: Add heterogenous sizes
        return pa.array([[4], [5], [6], [7], [8], [9]], type=pa_type)
    elif pa.types.is_struct(pa_type):
        return pa.array(
            [{"v": 4}, {"v": 5}, {"v": 6}, {"v": 7}, {"v": 8}, {"v": 9}],
            type=pa_type,
        )
    raise ValueError("Unsupported type")


@pytest.fixture(scope="module")
def target_column(pa_target_column):
    return plc.interop.from_arrow(pa_target_column)


@pytest.fixture
def mutable_target_column(target_column):
    return target_column.copy()


@pytest.fixture(scope="module")
def pa_source_table(pa_input_column):
    return pa.table([pa_input_column] * 3, [""] * 3)


@pytest.fixture(scope="module")
def source_table(pa_source_table):
    return plc.interop.from_arrow(pa_source_table)


@pytest.fixture(scope="module")
def pa_target_table(pa_target_column):
    return pa.table([pa_target_column] * 3, [""] * 3)


@pytest.fixture(scope="module")
def target_table(pa_target_table):
    return plc.interop.from_arrow(pa_target_table)


@pytest.fixture(scope="module")
def pa_source_scalar(pa_type):
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        return pa.scalar(1, type=pa_type)
    elif pa.types.is_string(pa_type):
        return pa.scalar("a", type=pa_type)
    elif pa.types.is_boolean(pa_type):
        return pa.scalar(False, type=pa_type)
    elif pa.types.is_list(pa_type):
        # TODO: Longer list?
        return pa.scalar([1], type=pa_type)
    elif pa.types.is_struct(pa_type):
        return pa.scalar({"v": 1}, type=pa_type)
    raise ValueError("Unsupported type")


@pytest.fixture(scope="module")
def source_scalar(pa_source_scalar):
    return plc.interop.from_arrow(pa_source_scalar)


@pytest.fixture(scope="module")
def pa_mask(pa_target_column):
    return pa.array([True, False] * (len(pa_target_column) // 2))


@pytest.fixture(scope="module")
def mask(pa_mask):
    return plc.interop.from_arrow(pa_mask)


def test_gather(target_table, pa_target_table, index_column, pa_index_column):
    result = plc.copying.gather(
        target_table,
        index_column,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    )
    expected = pa_target_table.take(pa_index_column)
    assert_table_eq(result, expected)


def test_gather_map_has_nulls(target_table):
    gather_map = plc.interop.from_arrow(pa.array([0, 1, None]))
    with cudf_raises(ValueError):
        plc.copying.gather(
            target_table,
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        )


def _pyarrow_index_to_mask(indices, mask_size):
    # Convert a list of indices to a boolean mask.
    return pc.is_in(pa.array(range(mask_size)), pa.array(indices))


def _pyarrow_boolean_mask_scatter_column(source, mask, target):
    if isinstance(source, pa.Scalar):
        # if_else requires array lengths to match exactly or the replacement must be a
        # scalar, so we use this in the scalar case.
        return pc.if_else(mask, target, source)

    if isinstance(source, pa.ChunkedArray):
        source = source.combine_chunks()
    if isinstance(target, pa.ChunkedArray):
        target = target.combine_chunks()

    # replace_with_mask accepts a column whose size is the number of true values in
    # the mask, so we can use it for columnar scatters.
    return pc.replace_with_mask(target, mask, source)


def _pyarrow_boolean_mask_scatter_table(source, mask, target_table):
    # pyarrow equivalent of cudf's boolean_mask_scatter.
    return pa.table(
        [
            _pyarrow_boolean_mask_scatter_column(r, mask, v)
            for v, r in zip(target_table, source)
        ],
        [""] * target_table.num_columns,
    )


def test_scatter_table(
    source_table,
    pa_source_table,
    index_column,
    pa_index_column,
    target_table,
    pa_target_table,
):
    result = plc.copying.scatter(
        source_table,
        index_column,
        target_table,
    )

    if pa.types.is_list(
        dtype := pa_target_table[0].type
    ) or pa.types.is_struct(dtype):
        # pyarrow does not support scattering with list data. If and when they do,
        # replace this hardcoding with their implementation.
        with pytest.raises(pa.ArrowNotImplementedError):
            _pyarrow_boolean_mask_scatter_table(
                pa_source_table,
                _pyarrow_index_to_mask(
                    pa_index_column, pa_target_table.num_rows
                ),
                pa_target_table,
            )

        if pa.types.is_list(dtype := pa_target_table[0].type):
            expected = pa.table(
                [pa.array([[4], [1], [2], [3], [8], [9]])] * 3, [""] * 3
            )
        elif pa.types.is_struct(dtype):
            expected = pa.table(
                [
                    pa.array(
                        [
                            {"v": 4},
                            {"v": 1},
                            {"v": 2},
                            {"v": 3},
                            {"v": 8},
                            {"v": 9},
                        ],
                        type=DEFAULT_STRUCT_TESTING_TYPE,
                    )
                ]
                * 3,
                [""] * 3,
            )
    else:
        expected = _pyarrow_boolean_mask_scatter_table(
            pa_source_table,
            _pyarrow_index_to_mask(pa_index_column, pa_target_table.num_rows),
            pa_target_table,
        )

    assert_table_eq(result, expected)


def test_scatter_table_num_col_mismatch(
    source_table, index_column, target_table
):
    # Number of columns in source and target must match.
    with cudf_raises(ValueError):
        plc.copying.scatter(
            plc.Table(source_table.columns()[:2]),
            index_column,
            target_table,
        )


def test_scatter_table_num_row_mismatch(source_table, target_table):
    # Number of rows in source and scatter map must match.
    with cudf_raises(ValueError):
        plc.copying.scatter(
            source_table,
            plc.interop.from_arrow(
                pa.array(range(source_table.num_rows() * 2))
            ),
            target_table,
        )


def test_scatter_table_map_has_nulls(source_table, target_table):
    with cudf_raises(ValueError):
        plc.copying.scatter(
            source_table,
            plc.interop.from_arrow(pa.array([None] * source_table.num_rows())),
            target_table,
        )


def test_scatter_table_type_mismatch(source_table, index_column, target_table):
    with cudf_raises(TypeError):
        if is_integer(
            dtype := target_table.columns()[0].type()
        ) or is_floating(dtype):
            pa_array = pa.array([True] * source_table.num_rows())
        else:
            pa_array = pa.array([1] * source_table.num_rows())
        ncol = source_table.num_columns()
        pa_table = pa.table([pa_array] * ncol, [""] * ncol)
        plc.copying.scatter(
            plc.interop.from_arrow(pa_table),
            index_column,
            target_table,
        )


def test_scatter_scalars(
    source_scalar,
    pa_source_scalar,
    index_column,
    pa_index_column,
    target_table,
    pa_target_table,
):
    result = plc.copying.scatter(
        [source_scalar] * target_table.num_columns(),
        index_column,
        target_table,
    )

    expected = _pyarrow_boolean_mask_scatter_table(
        [pa_source_scalar] * target_table.num_columns(),
        pc.invert(
            _pyarrow_index_to_mask(pa_index_column, pa_target_table.num_rows)
        ),
        pa_target_table,
    )

    assert_table_eq(result, expected)


def test_scatter_scalars_num_scalars_mismatch(
    source_scalar, index_column, target_table
):
    with cudf_raises(ValueError):
        plc.copying.scatter(
            [source_scalar] * (target_table.num_columns() - 1),
            index_column,
            target_table,
        )


def test_scatter_scalars_map_has_nulls(source_scalar, target_table):
    with cudf_raises(ValueError):
        plc.copying.scatter(
            [source_scalar] * target_table.num_columns(),
            plc.interop.from_arrow(pa.array([None, None])),
            target_table,
        )


def test_scatter_scalars_type_mismatch(index_column, target_table):
    with cudf_raises(TypeError):
        if is_integer(
            dtype := target_table.columns()[0].type()
        ) or is_floating(dtype):
            source_scalar = [plc.interop.from_arrow(pa.scalar(True))]
        else:
            source_scalar = [plc.interop.from_arrow(pa.scalar(1))]
        plc.copying.scatter(
            source_scalar * target_table.num_columns(),
            index_column,
            target_table,
        )


def test_empty_like_column(input_column):
    result = plc.copying.empty_like(input_column)
    assert result.type() == input_column.type()


def test_empty_like_table(source_table):
    result = plc.copying.empty_like(source_table)
    assert result.num_columns() == source_table.num_columns()
    for icol, rcol in zip(source_table.columns(), result.columns()):
        assert rcol.type() == icol.type()


@pytest.mark.parametrize("size", [None, 10])
def test_allocate_like(input_column, size):
    if is_fixed_width(input_column.type()):
        result = plc.copying.allocate_like(
            input_column, plc.copying.MaskAllocationPolicy.RETAIN, size=size
        )
        assert result.type() == input_column.type()
        assert result.size() == (input_column.size() if size is None else size)
    else:
        with pytest.raises(TypeError):
            plc.copying.allocate_like(
                input_column,
                plc.copying.MaskAllocationPolicy.RETAIN,
                size=size,
            )


def test_copy_range_in_place(
    input_column, pa_input_column, mutable_target_column, pa_target_column
):
    if not is_fixed_width(mutable_target_column.type()):
        with pytest.raises(TypeError):
            plc.copying.copy_range_in_place(
                input_column,
                mutable_target_column,
                0,
                input_column.size(),
                0,
            )
    else:
        plc.copying.copy_range_in_place(
            input_column,
            mutable_target_column,
            0,
            input_column.size(),
            0,
        )
        expected = _pyarrow_boolean_mask_scatter_column(
            pa_input_column,
            _pyarrow_index_to_mask(
                range(len(pa_input_column)), len(pa_target_column)
            ),
            pa_target_column,
        )
        assert_column_eq(mutable_target_column, expected)


def test_copy_range_in_place_out_of_bounds(
    input_column, mutable_target_column
):
    if is_fixed_width(mutable_target_column.type()):
        with cudf_raises(IndexError):
            plc.copying.copy_range_in_place(
                input_column,
                mutable_target_column,
                5,
                5 + input_column.size(),
                0,
            )


def test_copy_range_in_place_different_types(mutable_target_column):
    if is_integer(dtype := mutable_target_column.type()) or is_floating(dtype):
        input_column = plc.interop.from_arrow(pa.array(["a", "b", "c"]))
    else:
        input_column = plc.interop.from_arrow(pa.array([1, 2, 3]))

    with cudf_raises(TypeError):
        plc.copying.copy_range_in_place(
            input_column,
            mutable_target_column,
            0,
            input_column.size(),
            0,
        )


def test_copy_range_in_place_null_mismatch(
    pa_input_column, mutable_target_column
):
    if is_fixed_width(mutable_target_column.type()):
        pa_input_column = pc.if_else(
            _pyarrow_index_to_mask([0], len(pa_input_column)),
            pa_input_column,
            pa.scalar(None, type=pa_input_column.type),
        )
        input_column = plc.interop.from_arrow(pa_input_column)
        with cudf_raises(ValueError):
            plc.copying.copy_range_in_place(
                input_column,
                mutable_target_column,
                0,
                input_column.size(),
                0,
            )


def test_copy_range(
    input_column, pa_input_column, target_column, pa_target_column
):
    if is_fixed_width(dtype := target_column.type()) or is_string(dtype):
        result = plc.copying.copy_range(
            input_column,
            target_column,
            0,
            input_column.size(),
            0,
        )
        expected = _pyarrow_boolean_mask_scatter_column(
            pa_input_column,
            _pyarrow_index_to_mask(
                range(len(pa_input_column)), len(pa_target_column)
            ),
            pa_target_column,
        )
        assert_column_eq(result, expected)
    else:
        with pytest.raises(TypeError):
            plc.copying.copy_range(
                input_column,
                target_column,
                0,
                input_column.size(),
                0,
            )


def test_copy_range_out_of_bounds(input_column, target_column):
    with cudf_raises(IndexError):
        plc.copying.copy_range(
            input_column,
            target_column,
            5,
            5 + input_column.size(),
            0,
        )


def test_copy_range_different_types(target_column):
    if is_integer(dtype := target_column.type()) or is_floating(dtype):
        input_column = plc.interop.from_arrow(pa.array(["a", "b", "c"]))
    else:
        input_column = plc.interop.from_arrow(pa.array([1, 2, 3]))

    with cudf_raises(TypeError):
        plc.copying.copy_range(
            input_column,
            target_column,
            0,
            input_column.size(),
            0,
        )


def test_shift(
    target_column, pa_target_column, source_scalar, pa_source_scalar
):
    shift = 2
    if is_fixed_width(dtype := target_column.type()) or is_string(dtype):
        result = plc.copying.shift(target_column, shift, source_scalar)
        expected = pa.concat_arrays(
            [pa.array([pa_source_scalar] * shift), pa_target_column[:-shift]]
        )
        assert_column_eq(result, expected)
    else:
        with pytest.raises(TypeError):
            plc.copying.shift(target_column, shift, source_scalar)


def test_shift_type_mismatch(target_column):
    if is_integer(dtype := target_column.type()) or is_floating(dtype):
        fill_value = plc.interop.from_arrow(pa.scalar("a"))
    else:
        fill_value = plc.interop.from_arrow(pa.scalar(1))

    with cudf_raises(TypeError):
        plc.copying.shift(target_column, 2, fill_value)


def test_slice_column(target_column, pa_target_column):
    bounds = list(range(6))
    upper_bounds = bounds[1::2]
    lower_bounds = bounds[::2]
    result = plc.copying.slice(target_column, bounds)
    for lb, ub, slice_ in zip(lower_bounds, upper_bounds, result):
        assert_column_eq(slice_, pa_target_column[lb:ub])


def test_slice_column_wrong_length(target_column):
    with cudf_raises(ValueError):
        plc.copying.slice(target_column, list(range(5)))


def test_slice_column_decreasing(target_column):
    with cudf_raises(ValueError):
        plc.copying.slice(target_column, list(range(5, -1, -1)))


def test_slice_column_out_of_bounds(target_column):
    with cudf_raises(IndexError):
        plc.copying.slice(target_column, list(range(2, 8)))


def test_slice_table(target_table, pa_target_table):
    bounds = list(range(6))
    upper_bounds = bounds[1::2]
    lower_bounds = bounds[::2]
    result = plc.copying.slice(target_table, bounds)
    for lb, ub, slice_ in zip(lower_bounds, upper_bounds, result):
        assert_table_eq(slice_, pa_target_table[lb:ub])


def test_split_column(target_column, pa_target_column):
    upper_bounds = [1, 3, 5]
    lower_bounds = [0] + upper_bounds[:-1]
    result = plc.copying.split(target_column, upper_bounds)
    for lb, ub, split in zip(lower_bounds, upper_bounds, result):
        assert_column_eq(split, pa_target_column[lb:ub])


def test_split_column_decreasing(target_column):
    with cudf_raises(ValueError):
        plc.copying.split(target_column, list(range(5, -1, -1)))


def test_split_column_out_of_bounds(target_column):
    with cudf_raises(IndexError):
        plc.copying.split(target_column, list(range(5, 8)))


def test_split_table(target_table, pa_target_table):
    upper_bounds = [1, 3, 5]
    lower_bounds = [0] + upper_bounds[:-1]
    result = plc.copying.split(target_table, upper_bounds)
    for lb, ub, split in zip(lower_bounds, upper_bounds, result):
        assert_table_eq(split, pa_target_table[lb:ub])


def test_copy_if_else_column_column(
    target_column, pa_target_column, pa_source_scalar, mask, pa_mask
):
    pa_other_column = pa.concat_arrays(
        [pa.array([pa_source_scalar] * 2), pa_target_column[:-2]]
    )
    other_column = plc.interop.from_arrow(pa_other_column)

    result = plc.copying.copy_if_else(
        target_column,
        other_column,
        mask,
    )

    expected = pc.if_else(
        pa_mask,
        pa_target_column,
        pa_other_column,
    )
    assert_column_eq(result, expected)


def test_copy_if_else_wrong_type(target_column, mask):
    if is_integer(dtype := target_column.type()) or is_floating(dtype):
        input_column = plc.interop.from_arrow(
            pa.array(["a"] * target_column.size())
        )
    else:
        input_column = plc.interop.from_arrow(
            pa.array([1] * target_column.size())
        )

    with cudf_raises(TypeError):
        plc.copying.copy_if_else(input_column, target_column, mask)


def test_copy_if_else_wrong_type_mask(target_column):
    with cudf_raises(TypeError):
        plc.copying.copy_if_else(
            target_column,
            target_column,
            plc.interop.from_arrow(
                pa.array([1.0, 2.0] * (target_column.size() // 2))
            ),
        )


def test_copy_if_else_wrong_size(target_column):
    with cudf_raises(ValueError):
        plc.copying.copy_if_else(
            plc.interop.from_arrow(pa.array([1])),
            target_column,
            plc.interop.from_arrow(
                pa.array([True, False] * (target_column.size() // 2))
            ),
        )


def test_copy_if_else_wrong_size_mask(target_column):
    with cudf_raises(ValueError):
        plc.copying.copy_if_else(
            target_column,
            target_column,
            plc.interop.from_arrow(pa.array([True])),
        )


@pytest.mark.parametrize("array_left", [True, False])
def test_copy_if_else_column_scalar(
    target_column,
    pa_target_column,
    source_scalar,
    pa_source_scalar,
    array_left,
    mask,
    pa_mask,
):
    args = (
        (target_column, source_scalar)
        if array_left
        else (source_scalar, target_column)
    )
    result = plc.copying.copy_if_else(
        *args,
        mask,
    )

    pa_args = (
        (pa_target_column, pa_source_scalar)
        if array_left
        else (pa_source_scalar, pa_target_column)
    )
    expected = pc.if_else(
        pa_mask,
        *pa_args,
    )
    assert_column_eq(result, expected)


def test_boolean_mask_scatter_from_table(
    source_table,
    pa_source_table,
    target_table,
    pa_target_table,
    mask,
    pa_mask,
):
    result = plc.copying.boolean_mask_scatter(
        source_table,
        target_table,
        mask,
    )

    if pa.types.is_list(
        dtype := pa_target_table[0].type
    ) or pa.types.is_struct(dtype):
        # pyarrow does not support scattering with list data. If and when they do,
        # replace this hardcoding with their implementation.
        with pytest.raises(pa.ArrowNotImplementedError):
            _pyarrow_boolean_mask_scatter_table(
                pa_source_table, pa_mask, pa_target_table
            )

        if pa.types.is_list(dtype := pa_target_table[0].type):
            expected = pa.table(
                [pa.array([[1], [5], [2], [7], [3], [9]])] * 3, [""] * 3
            )
        elif pa.types.is_struct(dtype):
            expected = pa.table(
                [
                    pa.array(
                        [
                            {"v": 1},
                            {"v": 5},
                            {"v": 2},
                            {"v": 7},
                            {"v": 3},
                            {"v": 9},
                        ],
                        type=DEFAULT_STRUCT_TESTING_TYPE,
                    )
                ]
                * 3,
                [""] * 3,
            )
    else:
        expected = _pyarrow_boolean_mask_scatter_table(
            pa_source_table, pa_mask, pa_target_table
        )

    assert_table_eq(result, expected)


def test_boolean_mask_scatter_from_wrong_num_cols(source_table, target_table):
    with cudf_raises(ValueError):
        plc.copying.boolean_mask_scatter(
            plc.Table(source_table.columns()[:2]),
            target_table,
            plc.interop.from_arrow(pa.array([True, False] * 3)),
        )


def test_boolean_mask_scatter_from_wrong_mask_size(source_table, target_table):
    with cudf_raises(ValueError):
        plc.copying.boolean_mask_scatter(
            source_table,
            target_table,
            plc.interop.from_arrow(pa.array([True, False] * 2)),
        )


def test_boolean_mask_scatter_from_wrong_num_true(source_table, target_table):
    with cudf_raises(ValueError):
        plc.copying.boolean_mask_scatter(
            plc.Table(source_table.columns()[:2]),
            target_table,
            plc.interop.from_arrow(
                pa.array([True, False] * 2 + [False, False])
            ),
        )


def test_boolean_mask_scatter_from_wrong_col_type(target_table, mask):
    if is_integer(dtype := target_table.columns()[0].type()) or is_floating(
        dtype
    ):
        input_column = plc.interop.from_arrow(pa.array(["a", "b", "c"]))
    else:
        input_column = plc.interop.from_arrow(pa.array([1, 2, 3]))

    with cudf_raises(TypeError):
        plc.copying.boolean_mask_scatter(
            plc.Table([input_column] * 3), target_table, mask
        )


def test_boolean_mask_scatter_from_wrong_mask_type(source_table, target_table):
    with cudf_raises(TypeError):
        plc.copying.boolean_mask_scatter(
            source_table,
            target_table,
            plc.interop.from_arrow(pa.array([1.0, 2.0] * 3)),
        )


def test_boolean_mask_scatter_from_scalars(
    source_scalar,
    pa_source_scalar,
    target_table,
    pa_target_table,
    mask,
    pa_mask,
):
    result = plc.copying.boolean_mask_scatter(
        [source_scalar] * 3,
        target_table,
        mask,
    )

    expected = _pyarrow_boolean_mask_scatter_table(
        [pa_source_scalar] * target_table.num_columns(),
        pc.invert(pa_mask),
        pa_target_table,
    )

    assert_table_eq(result, expected)


def test_get_element(input_column, pa_input_column):
    index = 1
    result = plc.copying.get_element(input_column, index)

    assert (
        plc.interop.to_arrow(
            result, metadata_from_arrow_array(pa_input_column)
        ).as_py()
        == pa_input_column[index].as_py()
    )


def test_get_element_out_of_bounds(input_column):
    with cudf_raises(IndexError):
        plc.copying.get_element(input_column, 100)
