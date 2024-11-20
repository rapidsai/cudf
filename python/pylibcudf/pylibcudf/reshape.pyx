# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.reshape cimport (
    interleave_columns as cpp_interleave_columns,
    tile as cpp_tile,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .table cimport Table

__all__ = ["interleave_columns", "tile"]

cpdef Column interleave_columns(Table source_table):
    """Interleave columns of a table into a single column.

    Converts the column major table `input` into a row major column.

    Example:
    in     = [[A1, A2, A3], [B1, B2, B3]]
    return = [A1, B1, A2, B2, A3, B3]

    For details, see :cpp:func:`interleave_columns`.

    Parameters
    ----------
    source_table: Table
        The input table to interleave

    Returns
    -------
    Column
        A new column which is the result of interleaving the input columns
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_interleave_columns(source_table.view())

    return Column.from_libcudf(move(c_result))


cpdef Table tile(Table source_table, size_type count):
    """Repeats the rows from input table count times to form a new table.

    For details, see :cpp:func:`tile`.

    Parameters
    ----------
    source_table: Table
        The input table containing rows to be repeated
    count: size_type
        The number of times to tile "rows". Must be non-negative

    Returns
    -------
    Table
        The table containing the tiled "rows"
    """
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = cpp_tile(source_table.view(), count)

    return Table.from_libcudf(move(c_result))
