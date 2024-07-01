# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.table.table cimport table


cdef extern from "cudf/strings/split/partition.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[table] partition(
        column_view source_strings,
        string_scalar delimiter) except +

    cdef unique_ptr[table] rpartition(
        column_view source_strings,
        string_scalar delimiter) except +
