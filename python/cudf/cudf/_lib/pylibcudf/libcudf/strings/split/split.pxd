# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.regex_program cimport regex_program
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/split/split.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] split(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +

    cdef unique_ptr[table] rsplit(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +

    cdef unique_ptr[column] split_record(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +

    cdef unique_ptr[column] rsplit_record(
        column_view source_strings,
        string_scalar delimiter,
        size_type maxsplit) except +


cdef extern from "cudf/strings/split/split_re.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] split_re(
        const column_view& source_strings,
        regex_program,
        size_type maxsplit) except +

    cdef unique_ptr[table] rsplit_re(
        const column_view& source_strings,
        regex_program,
        size_type maxsplit) except +

    cdef unique_ptr[column] split_record_re(
        const column_view& source_strings,
        regex_program,
        size_type maxsplit) except +

    cdef unique_ptr[column] rsplit_record_re(
        const column_view& source_strings,
        regex_program,
        size_type maxsplit) except +
