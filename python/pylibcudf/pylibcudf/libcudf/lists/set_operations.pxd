# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport nan_equality, null_equality
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/lists/set_operations.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] difference_distinct(
        const lists_column_view& lhs,
        const lists_column_view& rhs,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] have_overlap(
        const lists_column_view& lhs,
        const lists_column_view& rhs,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] intersect_distinct(
        const lists_column_view& lhs,
        const lists_column_view& rhs,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] union_distinct(
        const lists_column_view& lhs,
        const lists_column_view& rhs,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
