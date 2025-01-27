# Copyright (c) 2019-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings.regex_program cimport regex_program


cdef extern from "cudf/strings/findall.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] findall(
        column_view input,
        regex_program prog) except +libcudf_exception_handler

    cdef unique_ptr[column] find_re(
        column_view input,
        regex_program prog) except +libcudf_exception_handler
