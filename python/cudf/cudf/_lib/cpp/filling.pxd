# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport (
    column_view,
    mutable_column_view
)
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/filling.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] fill(
        const column_view & input,
        size_type begin,
        size_type end,
        const scalar & value
    ) except +

    cdef void fill_in_place(
        const mutable_column_view & destination,
        size_type beign,
        size_type end,
        const scalar & value
    ) except +

    cdef unique_ptr[table] repeat(
        const table_view & input,
        const column_view & count,
        bool check_count
    ) except +

    cdef unique_ptr[table] repeat(
        const table_view & input,
        size_type count
    ) except +

    cdef unique_ptr[column] sequence(
        size_type size,
        const scalar & init,
        const scalar & step
    ) except +
