# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings.regex_program cimport regex_program
from pylibcudf.libcudf.table.table cimport table


cdef extern from "cudf/strings/extract.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[table] extract(
        column_view source_strings,
        regex_program) except +
