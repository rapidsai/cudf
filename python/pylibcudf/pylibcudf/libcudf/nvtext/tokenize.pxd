# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type


cdef extern from "nvtext/tokenize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] tokenize(
        const column_view & strings,
        const string_scalar & delimiter
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] tokenize(
        const column_view & strings,
        const column_view & delimiters
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] count_tokens(
        const column_view & strings,
        const string_scalar & delimiter
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] count_tokens(
        const column_view & strings,
        const column_view & delimiters
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] character_tokenize(
        const column_view & strings
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] detokenize(
        const column_view & strings,
        const column_view & row_indices,
        const string_scalar & separator
    ) except +libcudf_exception_handler

    cdef struct tokenize_vocabulary "nvtext::tokenize_vocabulary":
        pass

    cdef unique_ptr[tokenize_vocabulary] load_vocabulary(
        const column_view & strings
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] tokenize_with_vocabulary(
        const column_view & strings,
        const tokenize_vocabulary & vocabulary,
        const string_scalar & delimiter,
        size_type default_id
    ) except +libcudf_exception_handler
