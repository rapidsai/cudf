# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "nvtext/wordpiece_tokenize.hpp" namespace "nvtext" nogil:

    cdef struct wordpiece_vocabulary:
        pass

    cdef unique_ptr[wordpiece_vocabulary] load_wordpiece_vocabulary(
        const column_view & strings,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] wordpiece_tokenize(
        const column_view & strings,
        const wordpiece_vocabulary & vocabulary,
        size_type max_tokens_per_row,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
