# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.regex_program cimport regex_program
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/replace_re.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] replace_re(
        column_view source_strings,
        regex_program,
        string_scalar repl,
        size_type maxrepl) except +

    cdef unique_ptr[column] replace_with_backrefs(
        column_view source_strings,
        regex_program,
        string repl) except +

    cdef unique_ptr[column] replace_re(
        column_view source_strings,
        vector[string] patterns,
        column_view repls) except +
