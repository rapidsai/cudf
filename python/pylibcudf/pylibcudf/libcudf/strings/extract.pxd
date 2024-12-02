# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings.regex_program cimport regex_program
from pylibcudf.libcudf.table.table cimport table


cdef extern from "cudf/strings/extract.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[table] extract(
        column_view input,
        regex_program prog) except +libcudf_exception_handler

    cdef unique_ptr[column] extract_all_record(
        column_view input,
        regex_program prog) except +libcudf_exception_handler
