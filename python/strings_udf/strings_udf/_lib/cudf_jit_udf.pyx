# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import as_buffer

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column, column_view
from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from strings_udf._lib.cpp.strings_udf cimport (
    column_from_udf_string_array as cpp_column_from_udf_string_array,
    free_udf_string_array as cpp_free_udf_string_array,
    to_string_view_array as cpp_to_string_view_array,
    udf_string,
)


def column_to_string_view_array(Column strings_col):
    cdef unique_ptr[device_buffer] c_buffer
    cdef column_view input_view = strings_col.view()
    with nogil:
        c_buffer = move(cpp_to_string_view_array(input_view))

    device_buffer = DeviceBuffer.c_from_unique_ptr(move(c_buffer))
    return as_buffer(device_buffer)


def column_from_udf_string_array(DeviceBuffer d_buffer):
    cdef size_t size = int(d_buffer.c_size() / sizeof(udf_string))
    cdef udf_string* data = <udf_string*>d_buffer.c_data()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_column_from_udf_string_array(data, size))
        cpp_free_udf_string_array(data, size)

    result = Column.from_unique_ptr(move(c_result))

    return result
