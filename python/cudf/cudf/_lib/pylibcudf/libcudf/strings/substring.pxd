# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/slice.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        numeric_scalar[size_type] start,
        numeric_scalar[size_type] end,
        numeric_scalar[size_type] step) except +

    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        column_view starts,
        column_view stops) except +
