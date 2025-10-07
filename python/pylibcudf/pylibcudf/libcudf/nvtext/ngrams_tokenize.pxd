# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "nvtext/ngrams_tokenize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] ngrams_tokenize(
        const column_view & strings,
        size_type ngrams,
        const string_scalar & delimiter,
        const string_scalar & separator,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
