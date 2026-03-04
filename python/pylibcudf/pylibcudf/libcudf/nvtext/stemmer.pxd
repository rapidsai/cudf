# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "nvtext/stemmer.hpp" namespace "nvtext" nogil:
    cpdef enum class letter_type:
        CONSONANT
        VOWEL

    cdef unique_ptr[column] porter_stemmer_measure(
        const column_view & strings,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] is_letter(
        column_view source_strings,
        letter_type ltype,
        size_type character_index,
        cuda_stream_view stream) except +libcudf_exception_handler

    cdef unique_ptr[column] is_letter(
        column_view source_strings,
        letter_type ltype,
        column_view indices,
        cuda_stream_view stream) except +libcudf_exception_handler

ctypedef int32_t underlying_type_t_letter_type
