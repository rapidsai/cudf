# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t, uint16_t, uintptr_t

from cudf._lib.pylibcudf.libcudf.strings_udf cimport (
    get_character_cases_table as cpp_get_character_cases_table,
    get_character_flags_table as cpp_get_character_flags_table,
    get_special_case_mapping_table as cpp_get_special_case_mapping_table,
)

import numpy as np

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import as_buffer

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column, column_view
from cudf._lib.pylibcudf.libcudf.strings_udf cimport (
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

    db = DeviceBuffer.c_from_unique_ptr(move(c_buffer))
    return as_buffer(db, exposed=True)


def column_from_udf_string_array(DeviceBuffer d_buffer):
    cdef size_t size = int(d_buffer.c_size() / sizeof(udf_string))
    cdef udf_string* data = <udf_string*>d_buffer.c_data()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_column_from_udf_string_array(data, size))
        cpp_free_udf_string_array(data, size)

    result = Column.from_unique_ptr(move(c_result))

    return result


def get_character_flags_table_ptr():
    cdef const uint8_t* tbl_ptr = cpp_get_character_flags_table()
    return np.uintp(<uintptr_t>tbl_ptr)


def get_character_cases_table_ptr():
    cdef const uint16_t* tbl_ptr = cpp_get_character_cases_table()
    return np.uintp(<uintptr_t>tbl_ptr)


def get_special_case_mapping_table_ptr():
    cdef const void* tbl_ptr = cpp_get_special_case_mapping_table()
    return np.uintp(<uintptr_t>tbl_ptr)
