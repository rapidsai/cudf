# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from numba.np import numpy_support

import cudf
from cudf._lib.types import SUPPORTED_NUMPY_TO_LIBCUDF_TYPES
from cudf.core._internals.expressions import parse_expression
from cudf.core.buffer import acquire_spill_lock, as_buffer
from cudf.utils import cudautils

from cython.operator cimport dereference
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

cimport cudf._lib.pylibcudf.libcudf.transform as libcudf_transform
from cudf._lib.column cimport Column
from cudf._lib.expressions cimport Expression
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.expressions cimport expression
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport (
    bitmask_type,
    data_type,
    size_type,
    type_id,
)
from cudf._lib.types cimport underlying_type_t_type_id
from cudf._lib.utils cimport (
    columns_from_unique_ptr,
    data_from_table_view,
    table_view_from_columns,
)


@acquire_spill_lock()
def bools_to_mask(Column col):
    """
    Given an int8 (boolean) column, compress the data from booleans to bits and
    return a Buffer
    """
    cdef column_view col_view = col.view()
    cdef pair[unique_ptr[device_buffer], size_type] cpp_out
    cdef unique_ptr[device_buffer] up_db

    with nogil:
        cpp_out = move(libcudf_transform.bools_to_mask(col_view))
        up_db = move(cpp_out.first)

    rmm_db = DeviceBuffer.c_from_unique_ptr(move(up_db))
    buf = as_buffer(rmm_db)
    return buf


@acquire_spill_lock()
def mask_to_bools(object mask_buffer, size_type begin_bit, size_type end_bit):
    """
    Given a mask buffer, returns a boolean column representng bit 0 -> False
    and 1 -> True within range of [begin_bit, end_bit),
    """
    if not isinstance(mask_buffer, cudf.core.buffer.Buffer):
        raise TypeError("mask_buffer is not an instance of "
                        "cudf.core.buffer.Buffer")
    cdef bitmask_type* bit_mask = <bitmask_type*><uintptr_t>(
        mask_buffer.get_ptr(mode="read")
    )

    cdef unique_ptr[column] result
    with nogil:
        result = move(
            libcudf_transform.mask_to_bools(bit_mask, begin_bit, end_bit)
        )

    return Column.from_unique_ptr(move(result))


@acquire_spill_lock()
def nans_to_nulls(Column input):
    cdef column_view c_input = input.view()
    cdef pair[unique_ptr[device_buffer], size_type] c_output
    cdef unique_ptr[device_buffer] c_buffer

    with nogil:
        c_output = move(libcudf_transform.nans_to_nulls(c_input))
        c_buffer = move(c_output.first)

    if c_output.second == 0:
        return None

    return as_buffer(DeviceBuffer.c_from_unique_ptr(move(c_buffer)))


@acquire_spill_lock()
def transform(Column input, op):
    cdef column_view c_input = input.view()
    cdef string c_str
    cdef type_id c_tid
    cdef data_type c_dtype

    nb_type = numpy_support.from_dtype(input.dtype)
    nb_signature = (nb_type,)
    compiled_op = cudautils.compile_udf(op, nb_signature)
    c_str = compiled_op[0].encode('UTF-8')
    np_dtype = cudf.dtype(compiled_op[1])

    try:
        c_tid = <type_id> (
            <underlying_type_t_type_id> SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[
                np_dtype
            ]
        )
        c_dtype = data_type(c_tid)

    except KeyError:
        raise TypeError(
            "Result of window function has unsupported dtype {}"
            .format(np_dtype)
        )

    with nogil:
        c_output = move(libcudf_transform.transform(
            c_input,
            c_str,
            c_dtype,
            True
        ))

    return Column.from_unique_ptr(move(c_output))


def table_encode(list source_columns):
    cdef table_view c_input = table_view_from_columns(source_columns)
    cdef pair[unique_ptr[table], unique_ptr[column]] c_result

    with nogil:
        c_result = move(libcudf_transform.encode(c_input))

    return (
        columns_from_unique_ptr(move(c_result.first)),
        Column.from_unique_ptr(move(c_result.second))
    )


def one_hot_encode(Column input_column, Column categories):
    cdef column_view c_view_input = input_column.view()
    cdef column_view c_view_categories = categories.view()
    cdef pair[unique_ptr[column], table_view] c_result

    with nogil:
        c_result = move(
            libcudf_transform.one_hot_encode(c_view_input, c_view_categories)
        )

    # Notice, the data pointer of `owner` has been exposed
    # through `c_result.second` at this point.
    owner = Column.from_unique_ptr(
        move(c_result.first), data_ptr_exposed=True
    )

    pylist_categories = categories.to_arrow().to_pylist()
    encodings, _ = data_from_table_view(
        move(c_result.second),
        owner=owner,
        column_names=[
            x if x is not None else '<NA>' for x in pylist_categories
        ]
    )
    return encodings


@acquire_spill_lock()
def compute_column(list columns, tuple column_names, expr: str):
    """Compute a new column by evaluating an expression on a set of columns.

    Parameters
    ----------
    columns : list
        The set of columns forming the table to evaluate the expression on.
    column_names : tuple[str]
        The names associated with each column. These names are necessary to map
        column names in the expression to indices in the provided list of
        columns, which are what will be used by libcudf to evaluate the
        expression on the table.
    expr : str
        The expression to evaluate.
    """
    visitor = parse_expression(expr, column_names)

    # At the end, all the stack contains is the expression to evaluate.
    cdef Expression cudf_expr = visitor.expression
    cdef table_view tbl = table_view_from_columns(columns)
    cdef unique_ptr[column] col
    with nogil:
        col = move(
            libcudf_transform.compute_column(
                tbl,
                <expression &> dereference(cudf_expr.c_obj.get())
            )
        )
    return Column.from_unique_ptr(move(col))
