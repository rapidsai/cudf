# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from pylibcudf.libcudf.strings.regex_program cimport regex_program
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/strings/replace_re.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] replace_re(
        column_view input,
        regex_program prog,
        string_scalar replacement,
        size_type max_replace_count,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] replace_re(
        column_view input,
        vector[string] patterns,
        column_view replacements,
        regex_flags flags,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef unique_ptr[column] replace_with_backrefs(
        column_view input,
        regex_program prog,
        string replacement,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
