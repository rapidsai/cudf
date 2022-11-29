# Copyright (c) 2022, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t, uint16_t, uintptr_t

from strings_udf._lib.cpp.strings_udf cimport (
    get_character_cases_table as cpp_get_character_cases_table,
    get_character_flags_table as cpp_get_character_flags_table,
    get_special_case_mapping_table as cpp_get_special_case_mapping_table,
)

import numpy as np


def get_character_flags_table_ptr():
    cdef const uint8_t* tbl_ptr = cpp_get_character_flags_table()
    return np.uintp(<uintptr_t>tbl_ptr)


def get_character_cases_table_ptr():
    cdef const uint16_t* tbl_ptr = cpp_get_character_cases_table()
    return np.uintp(<uintptr_t>tbl_ptr)


def get_special_case_mapping_table_ptr():
    cdef const void* tbl_ptr = cpp_get_special_case_mapping_table()
    return np.uintp(<uintptr_t>tbl_ptr)
