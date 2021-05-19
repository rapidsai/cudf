# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.column.column cimport column

cdef extern from "cudf/strings/combine.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] concatenate(
        table_view source_strings,
        string_scalar separator,
        string_scalar narep) except +

    cdef unique_ptr[column] join_strings(
        column_view source_strings,
        string_scalar separator,
        string_scalar narep) except +

    cdef unique_ptr[column] concatenate_list_elements(
        column_view lists_strings_column,
        column_view separators,
        string_scalar separator_narep,
        string_scalar string_narep) except +

    cdef unique_ptr[column] concatenate_list_elements(
        column_view lists_strings_column,
        string_scalar separator,
        string_scalar narep) except +
