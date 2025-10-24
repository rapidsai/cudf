# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "nvtext/generate_ngrams.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] generate_ngrams(
        const column_view &strings,
        size_type ngrams,
        const string_scalar & separator,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] generate_character_ngrams(
        const column_view &strings,
        size_type ngrams,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] hash_character_ngrams(
        const column_view &strings,
        size_type ngrams,
        uint32_t seed,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
