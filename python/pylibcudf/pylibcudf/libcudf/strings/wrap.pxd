# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/wrap.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] wrap(
        column_view input,
        size_type width) except +
