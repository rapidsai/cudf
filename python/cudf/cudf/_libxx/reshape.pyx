# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table

from libcpp.memory cimport unique_ptr

from cudf._libxx.includes.reshape cimport (
    interleave_columns as cpp_interleave_columns,
    tile as cpp_tile
)


def interleave_columns(Table source_table):
    """
    Interleave columns of a table into a single column.

    Converts the column major table `input` into a row
    major column.

    Parameters
    ----------
    source_table : input Table containing columns to interleave.

    Example
    -------
    in     = [[A1, A2, A3], [B1, B2, B3]]
    return = [A1, B1, A2, B2, A3, B3]

    Returns
    -------
    The interleaved columns as a single column
    """
    
    cdef table_view c_view = source_table.data_view()

    with nogil:
        c_result = move(cpp_interleave_columns(c_view))

    return Column.from_unique_ptr(
        move(c_result)
    )


def tile(Table source_table, size_type count):
    """
    Repeats the rows from `input` table `count` times to
    form a new table.

    Parameters
    ----------
    source_table : input Table containing columns to
    interleave. count : Number of times to tile "rows".
    Must be non-negative.

    Example
    -------
    `output.num_columns() == input.num_columns()`
    `output.num_rows() == input.num_rows() * count`

    source_table  = [[8, 4, 7], [5, 2, 3]]
    count  = 2
    return = [[8, 4, 7, 8, 4, 7], [5, 2, 3, 5, 2, 3]]

    Returns
    -------
    The table containing the tiled "rows".
    """
    cdef size_type c_count = count
    cdef table_view c_view = source_table.view()

    with nogil:
        c_result = move(cpp_tile(c_view, c_count))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index
            is None else source_table._index_names)
    )
