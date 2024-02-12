# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column, column_view
from cudf._lib.cpp.concatenate cimport (
    concatenate_columns as libcudf_concatenate_columns,
    concatenate_tables as libcudf_concatenate_tables,
)
from cudf._lib.cpp.table.table cimport table, table_view
from cudf._lib.utils cimport (
    data_from_unique_ptr,
    make_column_views,
    table_view_from_table,
)

from cudf.core.buffer import acquire_spill_lock


@acquire_spill_lock()
def concat_columns(object columns):
    cdef unique_ptr[column] c_result
    cdef vector[column_view] c_views = make_column_views(columns)
    with nogil:
        c_result = move(libcudf_concatenate_columns(c_views))
    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def concat_tables(object tables, bool ignore_index=False):
    cdef unique_ptr[table] c_result
    cdef vector[table_view] c_views
    c_views.reserve(len(tables))
    for tbl in tables:
        c_views.push_back(table_view_from_table(tbl, ignore_index))
    with nogil:
        c_result = move(libcudf_concatenate_tables(c_views))

    return data_from_unique_ptr(
        move(c_result),
        column_names=tables[0]._column_names,
        index_names=None if ignore_index else tables[0]._index_names
    )
