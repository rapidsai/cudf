# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.column.column cimport column

cdef extern from "cudf/strings/combine.hpp" namespace "cudf::strings" nogil:

    ctypedef enum separator_on_nulls:
        YES 'cudf::strings::separator_on_nulls::YES'
        NO  'cudf::strings::separator_on_nulls::NO'

    ctypedef enum output_if_empty_list:
        EMPTY_STRING 'cudf::strings::output_if_empty_list::EMPTY_STRING'
        NULL_ELEMENT 'cudf::strings::output_if_empty_list::NULL_ELEMENT'

    cdef unique_ptr[column] concatenate(
        table_view source_strings,
        string_scalar separator,
        string_scalar narep) except +

    cdef unique_ptr[column] join_strings(
        column_view source_strings,
        string_scalar separator,
        string_scalar narep) except +

    cdef unique_ptr[column] join_list_elements(
        column_view lists_strings_column,
        column_view separators,
        string_scalar separator_narep,
        string_scalar string_narep,
        separator_on_nulls separate_nulls,
        output_if_empty_list empty_list_policy) except +

    cdef unique_ptr[column] join_list_elements(
        column_view lists_strings_column,
        string_scalar separator,
        string_scalar narep,
        separator_on_nulls separate_nulls,
        output_if_empty_list empty_list_policy) except +
