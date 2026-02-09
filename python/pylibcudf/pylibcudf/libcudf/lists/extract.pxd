# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/lists/extract.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] extract_list_element(
        const lists_column_view&,
        size_type,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[column] extract_list_element(
        const lists_column_view&,
        const column_view&,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
