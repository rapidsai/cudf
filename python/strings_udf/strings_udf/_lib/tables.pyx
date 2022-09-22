# Copyright (c) 2022, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t, uintptr_t

from strings_udf._lib.cpp.strings_udf cimport (
    get_character_flags_table as cpp_get_character_flags_table,
)

import numpy as np


def get_character_flags_table_ptr():
    cdef const uint8_t* tbl_ptr = cpp_get_character_flags_table()
    return np.int64(<uintptr_t>tbl_ptr)
