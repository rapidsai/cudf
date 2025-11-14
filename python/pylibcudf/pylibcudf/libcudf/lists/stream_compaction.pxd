# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.stream_compaction cimport duplicate_keep_option
from pylibcudf.libcudf.types cimport nan_equality, null_equality
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/lists/stream_compaction.hpp" \
        namespace "cudf::lists" nogil:
    cdef unique_ptr[column] apply_boolean_mask(
        const lists_column_view& lists_column,
        const lists_column_view& boolean_mask,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] distinct(
        const lists_column_view& lists_column,
        null_equality nulls_equal,
        nan_equality nans_equal,
        duplicate_keep_option keep_option,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
