# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cudf/join.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[column], unique_ptr[column]] inner_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on
    ) except +
    cdef pair[unique_ptr[column], unique_ptr[column]] left_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on
    ) except +
    cdef pair[unique_ptr[column], unique_ptr[column]] full_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on
    ) except +
    cdef unique_ptr[table] left_semi_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[int] return_columns
    ) except +
    cdef unique_ptr[table] left_anti_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[int] return_columns
    ) except +
