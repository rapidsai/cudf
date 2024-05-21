# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/wrap.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] wrap(
        column_view source_strings,
        size_type width) except +
