# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t, uint16_t, uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings_udf cimport (
    column_from_udf_string_array as cpp_column_from_udf_string_array,
    free_udf_string_array as cpp_free_udf_string_array,
    get_character_cases_table as cpp_get_character_cases_table,
    get_character_flags_table as cpp_get_character_flags_table,
    get_cuda_build_version as cpp_get_cuda_build_version,
    get_special_case_mapping_table as cpp_get_special_case_mapping_table,
    to_string_view_array as cpp_to_string_view_array,
    udf_string,
)
from pylibcudf.libcudf.types cimport size_type

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer


cpdef int get_cuda_build_version():
    return cpp_get_cuda_build_version()

cpdef DeviceBuffer column_to_string_view_array(Column input):
    cdef unique_ptr[device_buffer] c_buffer
    with nogil:
        c_buffer = cpp_to_string_view_array(input.view())

    return DeviceBuffer.c_from_unique_ptr(move(c_buffer))

cdef Column column_from_udf_string_array(udf_string* input, size_type size):
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_column_from_udf_string_array(
            input,
            size
        )
        cpp_free_udf_string_array(input, size)

    return Column.from_libcudf(move(c_result))

cdef void free_udf_string_array(udf_string* input, size_type size):
    with nogil:
        cpp_free_udf_string_array(
            input,
            size
        )

cpdef uintptr_t get_character_flags_table():
    cdef const uint8_t* tbl_ptr
    with nogil:
        tbl_ptr = cpp_get_character_flags_table()
    return <uintptr_t>tbl_ptr

cpdef uintptr_t get_character_cases_table():
    cdef const uint16_t* tbl_ptr
    with nogil:
        tbl_ptr = cpp_get_character_cases_table()
    return <uintptr_t>tbl_ptr


cpdef uintptr_t get_special_case_mapping_table():
    cdef const void* tbl_ptr
    with nogil:
        tbl_ptr = cpp_get_special_case_mapping_table()
    return <uintptr_t>tbl_ptr
