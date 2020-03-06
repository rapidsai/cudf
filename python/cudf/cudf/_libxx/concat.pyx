# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.vector cimport vector

from cudf._libxx.cpp.concatenate cimport (
    concatenate_masks as libcudf_concatenate_masks,
    concatenate_columns as libcudf_concatenate_columns,
    concatenate_tables as libcudf_concatenate_tables
)
from cudf._libxx.cpp.column.column cimport column, column_view
from cudf._libxx.cpp.table.table cimport table, table_view

from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move

from cudf.core.buffer import Buffer

from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer

cpdef concat_masks(columns):
    cdef device_buffer c_result
    cdef unique_ptr[device_buffer] c_unique_result
    cdef vector[column_view] c_views = make_column_views(columns)
    with nogil:
        c_result = move(libcudf_concatenate_masks(c_views))
        c_unique_result = make_unique[device_buffer](move(c_result))
    return Buffer(DeviceBuffer.c_from_unique_ptr(move(c_unique_result)))


cpdef concat_columns(columns):
    cdef unique_ptr[column] c_result
    cdef vector[column_view] c_views = make_column_views(columns)
    with nogil:
        c_result = move(libcudf_concatenate_columns(c_views))
    return Column.from_unique_ptr(move(c_result))


cpdef concat_tables(tables):
    cdef unique_ptr[table] c_result
    cdef vector[table_view] c_views = make_table_views(tables)
    with nogil:
        c_result = move(libcudf_concatenate_tables(c_views))
    return Table.from_unique_ptr(
        move(c_result),
        column_names=tables[0]._column_names
    )


cdef vector[column_view] make_column_views(columns):
    cdef vector[column_view] views
    views.reserve(len(columns))
    for col in columns:
        views.push_back((<Column> col).view())
    return views


cdef vector[table_view] make_table_views(tables):
    cdef vector[table_view] views
    views.reserve(len(tables))
    for tbl in tables:
        views.push_back((<Table> tbl).data_view())
    return views
