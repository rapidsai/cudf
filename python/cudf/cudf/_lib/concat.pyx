# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.vector cimport vector
from libcpp.utility cimport move

from cudf._lib.cpp.concatenate cimport (
    concatenate_masks as libcudf_concatenate_masks,
    concatenate_columns as libcudf_concatenate_columns,
    concatenate_tables as libcudf_concatenate_tables
)
from cudf._lib.cpp.column.column cimport column, column_view
from cudf._lib.cpp.table.table cimport table, table_view

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table
from cudf._lib.utils cimport (
    make_column_views,
    make_table_views,
    make_table_data_views
)

from cudf.core.buffer import Buffer

from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer

cpdef concat_masks(object columns):
    cdef device_buffer c_result
    cdef unique_ptr[device_buffer] c_unique_result
    cdef vector[column_view] c_views = make_column_views(columns)
    with nogil:
        c_result = move(libcudf_concatenate_masks(c_views))
        c_unique_result = make_unique[device_buffer](move(c_result))
    return Buffer(DeviceBuffer.c_from_unique_ptr(move(c_unique_result)))


cpdef concat_columns(object columns):
    cdef unique_ptr[column] c_result
    cdef vector[column_view] c_views = make_column_views(columns)
    with nogil:
        c_result = move(libcudf_concatenate_columns(c_views))
    return Column.from_unique_ptr(move(c_result))


cpdef concat_tables(object tables, bool ignore_index=False):
    cdef unique_ptr[table] c_result
    cdef vector[table_view] c_views
    if ignore_index is False:
        c_views = make_table_views(tables)
    else:
        c_views = make_table_data_views(tables)
    with nogil:
        c_result = move(libcudf_concatenate_tables(c_views))
    return Table.from_unique_ptr(
        move(c_result),
        column_names=tables[0]._column_names,
        index_names=None if ignore_index else tables[0]._index_names
    )
