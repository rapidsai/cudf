# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/strings/slice.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        numeric_scalar[size_type] start,
        numeric_scalar[size_type] end,
        numeric_scalar[size_type] step) except +libcudf_exception_handler

    cdef unique_ptr[column] slice_strings(
        column_view source_strings,
        column_view starts,
        column_view stops) except +libcudf_exception_handler
