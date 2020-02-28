# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *

cdef extern from "cudf/transpose.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] transpose(table_view input_table) except +
