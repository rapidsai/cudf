# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.table.table cimport table


cdef extern from "cudf/strings/split/partition.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] partition(
        column_view input,
        string_scalar delimiter) except +libcudf_exception_handler

    cdef unique_ptr[table] rpartition(
        column_view input,
        string_scalar delimiter) except +libcudf_exception_handler
