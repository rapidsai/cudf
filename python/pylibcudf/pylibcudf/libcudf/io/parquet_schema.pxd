# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/io/parquet_schema.hpp" namespace "cudf::io::parquet" nogil:
    cdef cppclass FileMetaData:
        FileMetaData() except +libcudf_exception_handler
        int32_t version
        int64_t num_rows
        string created_by
