# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cudf._lib.cpp.types cimport char_utf8
from cudf._lib.cpp.scalar.scalar cimport string_scalar

cdef extern from "cudf/strings/translate.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] translate(
        column_view source_strings,
        vector[pair[char_utf8, char_utf8]] chars_table) except +

    cdef unique_ptr[column] filter_characters(
        column_view source_strings,
        vector[pair[char_utf8, char_utf8]] chars_table,
        bool keep,
        string_scalar replacement) except +
