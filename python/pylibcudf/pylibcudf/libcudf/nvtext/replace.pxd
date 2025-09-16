# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "nvtext/replace.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] replace_tokens(
        const column_view & strings,
        const column_view & targets,
        const column_view & replacements,
        const string_scalar & delimiter,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] filter_tokens(
        const column_view & strings,
        size_type min_token_length,
        const string_scalar & replacement,
        const string_scalar & delimiter,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
