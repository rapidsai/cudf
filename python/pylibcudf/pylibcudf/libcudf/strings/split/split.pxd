# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.regex_program cimport regex_program
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/split/split.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] split(
        column_view strings_column,
        string_scalar delimiter,
        size_type maxsplit) except +libcudf_exception_handler

    cdef unique_ptr[table] rsplit(
        column_view strings_column,
        string_scalar delimiter,
        size_type maxsplit) except +libcudf_exception_handler

    cdef unique_ptr[column] split_record(
        column_view strings,
        string_scalar delimiter,
        size_type maxsplit) except +libcudf_exception_handler

    cdef unique_ptr[column] rsplit_record(
        column_view strings,
        string_scalar delimiter,
        size_type maxsplit) except +libcudf_exception_handler


cdef extern from "cudf/strings/split/split_re.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] split_re(
        const column_view& input,
        regex_program prog,
        size_type maxsplit) except +libcudf_exception_handler

    cdef unique_ptr[table] rsplit_re(
        const column_view& input,
        regex_program prog,
        size_type maxsplit) except +libcudf_exception_handler

    cdef unique_ptr[column] split_record_re(
        const column_view& input,
        regex_program prog,
        size_type maxsplit) except +libcudf_exception_handler

    cdef unique_ptr[column] rsplit_record_re(
        const column_view& input,
        regex_program prog,
        size_type maxsplit) except +libcudf_exception_handler
