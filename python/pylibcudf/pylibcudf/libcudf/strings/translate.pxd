# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport char_utf8
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/translate.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] translate(
        column_view input,
        vector[pair[char_utf8, char_utf8]] chars_table,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cpdef enum class filter_type(bool):
        KEEP
        REMOVE

    cdef unique_ptr[column] filter_characters(
        column_view input,
        vector[pair[char_utf8, char_utf8]] characters_to_filter,
        filter_type keep_characters,
        string_scalar replacement,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
