# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from pylibcudf.libcudf.strings.regex_program cimport regex_program
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/replace_re.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] replace_re(
        column_view input,
        regex_program prog,
        string_scalar replacement,
        size_type max_replace_count) except +

    cdef unique_ptr[column] replace_re(
        column_view input,
        vector[string] patterns,
        column_view replacements,
        regex_flags flags) except +

    cdef unique_ptr[column] replace_with_backrefs(
        column_view input,
        regex_program prog,
        string replacement) except +
