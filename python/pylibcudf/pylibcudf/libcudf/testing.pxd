# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.exception_handler cimport libcudf_exception_handler

from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf_test/default_stream.hpp" namespace "cudf::test" nogil:
    cdef cuda_stream_view get_default_stream() except +libcudf_exception_handler
