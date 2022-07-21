# Copyright (c) 2022, NVIDIA CORPORATION.
#
# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from strings_udf._lib.cpp.strings_udf cimport get_character_flags_table as cpp_get_character_flags_table
from libc.stdint cimport uintptr_t, uint8_t
import numpy as np


def get_character_flags_table_ptr():
    cdef const uint8_t* tbl_ptr = cpp_get_character_flags_table()
    return np.int64(<uintptr_t>tbl_ptr)
