# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/lists/count_elements.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] count_elements(
        const lists_column_view&,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
