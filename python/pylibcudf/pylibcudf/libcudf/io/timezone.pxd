# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table cimport table
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/timezone.hpp" namespace "cudf" nogil:
    unique_ptr[table] make_timezone_transition_table(
        optional[string] tzif_dir,
        string timezone_name,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
