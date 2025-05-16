# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.side_type cimport side_type


cdef extern from "cudf/strings/strip.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] strip(
        column_view input,
        side_type side,
        string_scalar to_strip) except +libcudf_exception_handler
