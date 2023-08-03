# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column, column_view
from cudf._lib.cpp.concatenate cimport (
    concatenate_columns as libcudf_concatenate_columns,
    concatenate_masks as libcudf_concatenate_masks,
    concatenate_tables as libcudf_concatenate_tables,
)
from cudf._lib.cpp.libcpp.memory cimport make_unique
from cudf._lib.cpp.table.table cimport table, table_view
from cudf._lib.utils cimport (
    data_from_unique_ptr,
    make_column_views,
    table_view_from_table,
)

from cudf.core.buffer import acquire_spill_lock, as_buffer

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer


cpdef concat_masks(object columns):
    cdef device_buffer c_result
    cdef unique_ptr[device_buffer] c_unique_result
    cdef vector[column_view] c_views = make_column_views(columns)
    with nogil:
        c_result = move(libcudf_concatenate_masks(c_views))
        c_unique_result = move(make_unique[device_buffer](move(c_result)))
    return as_buffer(
        DeviceBuffer.c_from_unique_ptr(move(c_unique_result))
    )


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
