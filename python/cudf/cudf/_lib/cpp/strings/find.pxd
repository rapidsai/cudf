# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type


cdef extern from "cudf/strings/find.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] contains(
        column_view source_strings,
        string_scalar target) except +

    cdef unique_ptr[column] contains(
        column_view source_strings,
        column_view target_strings) except +

    cdef unique_ptr[column] ends_with(
        column_view source_strings,
        string_scalar target) except +

    cdef unique_ptr[column] ends_with(
        column_view source_strings,
        column_view target_strings) except +

    cdef unique_ptr[column] starts_with(
        column_view source_strings,
        string_scalar target) except +

    cdef unique_ptr[column] starts_with(
        column_view source_strings,
        column_view target_strings) except +

    cdef unique_ptr[column] find(
        column_view source_strings,
        string_scalar target,
        size_type start,
        size_type stop) except +

    cdef unique_ptr[column] find(
        column_view source_strings,
        column_view target,
        size_type start) except +

    cdef unique_ptr[column] rfind(
        column_view source_strings,
        string_scalar target,
        size_type start,
        size_type stop) except +
