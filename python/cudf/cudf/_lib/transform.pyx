# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from numba.np import numpy_support

import cudf
from cudf.core.buffer import acquire_spill_lock, as_buffer
from cudf.utils import cudautils

from pylibcudf cimport transform as plc_transform
from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def bools_to_mask(Column col):
    """
    Given an int8 (boolean) column, compress the data from booleans to bits and
    return a Buffer
    """
    mask, _ = plc_transform.bools_to_mask(col.to_pylibcudf(mode="read"))
    return as_buffer(mask)


@acquire_spill_lock()
def mask_to_bools(object mask_buffer, size_type begin_bit, size_type end_bit):
    """
    Given a mask buffer, returns a boolean column representng bit 0 -> False
    and 1 -> True within range of [begin_bit, end_bit),
    """
    if not isinstance(mask_buffer, cudf.core.buffer.Buffer):
        raise TypeError("mask_buffer is not an instance of "
                        "cudf.core.buffer.Buffer")
    plc_column = plc_transform.mask_to_bools(
        mask_buffer.get_ptr(mode="read"), begin_bit, end_bit
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def nans_to_nulls(Column input):
    mask, _ = plc_transform.nans_to_nulls(
        input.to_pylibcudf(mode="read")
    )
    return as_buffer(mask)


@acquire_spill_lock()
def transform(Column input, op):
    nb_type = numpy_support.from_dtype(input.dtype)
    nb_signature = (nb_type,)
    compiled_op = cudautils.compile_udf(op, nb_signature)
    np_dtype = cudf.dtype(compiled_op[1])

    plc_column = plc_transform.transform(
        input.to_pylibcudf(mode="read"),
        compiled_op[0],
        plc.column._datatype_from_dtype_desc(np_dtype.str[1:]),
        True
    )
    return Column.from_pylibcudf(plc_column)


def table_encode(list source_columns):
    plc_table, plc_column = plc_transform.encode(
        plc.Table([col.to_pylibcudf(mode="read") for col in source_columns])
    )

    return (
        [Column.from_pylibcudf(col) for col in plc_table.columns()],
        Column.from_pylibcudf(plc_column)
    )


def one_hot_encode(Column input_column, Column categories):
    plc_table = plc_transform.one_hot_encode(
        input_column.to_pylibcudf(mode="read"),
        categories.to_pylibcudf(mode="read"),
    )
    result_columns = [
        Column.from_pylibcudf(col, data_ptr_exposed=True)
        for col in plc_table.columns()
    ]
    result_labels = [
        x if x is not None else '<NA>'
        for x in categories.to_arrow().to_pylist()
    ]
    return dict(zip(result_labels, result_columns))


@acquire_spill_lock()
def compute_column(list columns, tuple column_names, str expr):
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
    result = plc_transform.compute_column(
        plc.Table([col.to_pylibcudf(mode="read") for col in columns]),
        plc.expressions.to_expression(expr, column_names),
    )
    return Column.from_pylibcudf(result)
