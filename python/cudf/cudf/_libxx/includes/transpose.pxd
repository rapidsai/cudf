# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *

from cudf._libxx.includes.table.table cimport table
from cudf._libxx.includes.table.table_view cimport table_view


cdef extern from "cudf/transpose.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] transpose(table_view input_table) except +
