# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.side_type cimport side_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/strings/strip.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] strip(
        column_view input,
        side_type side,
        string_scalar to_strip,
        cuda_stream_view stream) except +libcudf_exception_handler
