# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "nvtext/tokenize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] tokenize(
        const column_view & strings,
        const string_scalar & delimiter,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] tokenize(
        const column_view & strings,
        const column_view & delimiters,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] count_tokens(
        const column_view & strings,
        const string_scalar & delimiter,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] count_tokens(
        const column_view & strings,
        const column_view & delimiters,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] character_tokenize(
        const column_view & strings,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] detokenize(
        const column_view & strings,
        const column_view & row_indices,
        const string_scalar & separator,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef struct tokenize_vocabulary:
        pass

    cdef unique_ptr[tokenize_vocabulary] load_vocabulary(
        const column_view & strings,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] tokenize_with_vocabulary(
        const column_view & strings,
        const tokenize_vocabulary & vocabulary,
        const string_scalar & delimiter,
        size_type default_id,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
