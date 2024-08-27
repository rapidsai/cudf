# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/strings/find_multiple.hpp" namespace "cudf::strings" \
        nogil:

    cdef unique_ptr[column] find_multiple(
        column_view source_strings,
        column_view targets) except +
