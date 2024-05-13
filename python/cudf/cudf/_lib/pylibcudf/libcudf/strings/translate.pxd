# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.types cimport char_utf8


cdef extern from "cudf/strings/translate.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] translate(
        column_view source_strings,
        vector[pair[char_utf8, char_utf8]] chars_table) except +

    ctypedef enum filter_type:
        KEEP 'cudf::strings::filter_type::KEEP',
        REMOVE 'cudf::strings::filter_type::REMOVE'

    cdef unique_ptr[column] filter_characters(
        column_view source_strings,
        vector[pair[char_utf8, char_utf8]] chars_table,
        filter_type keep,
        string_scalar replacement) except +
