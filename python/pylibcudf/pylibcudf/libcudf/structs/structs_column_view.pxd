# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/structs/structs_column_view.hpp" namespace "cudf" nogil:
    cdef cppclass structs_column_view(column_view):
        structs_column_view() except +libcudf_exception_handler
        structs_column_view(
            const structs_column_view& structs_column
        ) except +libcudf_exception_handler
        structs_column_view(
            const column_view& structs_column
        ) except +libcudf_exception_handler
        structs_column_view& operator=(
            const structs_column_view&
        ) except +libcudf_exception_handler
        column_view parent() except +libcudf_exception_handler
        column_view get_sliced_child(
            size_type index,
            cuda_stream_view stream
        ) except +libcudf_exception_handler
