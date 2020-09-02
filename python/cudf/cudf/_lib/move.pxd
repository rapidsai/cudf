# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t
from rmm._lib.device_buffer cimport device_buffer
from cudf._lib.cpp.types cimport (
    size_type,
)
from cudf._lib.cpp.aggregation cimport aggregation
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.groupby cimport (
    groups,
    aggregation_request,
    aggregation_result
)
from cudf._lib.cpp.table.table_view cimport table_view
from pyarrow.includes.libarrow cimport CMessageReader
from cudf._lib.cpp.io.orc cimport orc_reader_options, orc_writer_options
cimport cudf._lib.cpp.io.types as cudf_io_types


# Note: declaring `move()` with `except +` doesn't work.
#
# Consider:
#     cdef unique_ptr[int] x = move(y)
#
# If `move()` is declared with `except +`, the generated C++ code
# looks something like this:
#
#    std::unique_ptr<int>  __pyx_v_x;
#    std::unique_ptr<int>  __pyx_v_y;
#    std::unique_ptr<int>  __pyx_t_1;
#    try {
#      __pyx_t_1 = std::move(__pyx_v_y);
#    } catch(...) {
#      __Pyx_CppExn2PyErr();
#      __PYX_ERR(0, 8, __pyx_L1_error)
#    }
#    __pyx_v_x = __pyx_t_1;
#
# where the last statement will result in a compiler error
# (copying a unique_ptr).
#
cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[column] move(unique_ptr[column])
    cdef unique_ptr[table] move(unique_ptr[table])
    cdef unique_ptr[aggregation] move(unique_ptr[aggregation])
    cdef unique_ptr[CMessageReader] move(unique_ptr[CMessageReader])
    cdef unique_ptr[vector[uint8_t]] move(unique_ptr[vector[uint8_t]])
    cdef vector[unique_ptr[column]] move(vector[unique_ptr[column]])
    cdef vector[unique_ptr[table]] move(vector[unique_ptr[table]])
    cdef vector[column_view] move(vector[column_view])
    cdef vector[table_view] move(vector[table_view])
    cdef vector[uint8_t] move(vector[uint8_t])
    cdef pair[unique_ptr[table], vector[size_type]] move(
        pair[unique_ptr[table], vector[size_type]])
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
    cdef unique_ptr[scalar] move(unique_ptr[scalar])
    cdef pair[unique_ptr[device_buffer], size_type] move(
        pair[unique_ptr[device_buffer], size_type]
    )
    cdef column_contents move(column_contents)
    cdef groups move(groups)
    cdef aggregation_request move(aggregation_request)
    cdef aggregation_result move(aggregation_result)
    cdef pair[unique_ptr[table], vector[aggregation_result]] move(
        pair[unique_ptr[table], vector[aggregation_result]]
    )
    cdef cudf_io_types.source_info move(cudf_io_types.source_info)
    cdef cudf_io_types.table_with_metadata move(
        cudf_io_types.table_with_metadata)
    cdef pair[unique_ptr[column], table_view] move(
        pair[unique_ptr[column], table_view]
    )
    cdef orc_reader_options move(orc_reader_options)
    cdef orc_writer_options move(orc_writer_options)
