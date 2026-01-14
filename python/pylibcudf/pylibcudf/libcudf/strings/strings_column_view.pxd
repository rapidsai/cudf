# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view

from rmm.librmm.cuda_stream_view cimport cuda_stream_view

cdef extern from "cudf/strings/strings_column_view.hpp" namespace "cudf" nogil:
    cdef cppclass strings_column_view:
        strings_column_view(column_view) except +libcudf_exception_handler
        int64_t chars_size(cuda_stream_view) except +libcudf_exception_handler
