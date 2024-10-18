# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar


cdef extern from "cudf/scalar/scalar_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] make_string_scalar(
        const string & _string
    ) except +libcudf_exception_handler
    cdef unique_ptr[scalar] make_fixed_width_scalar[T](
        T value
    ) except +libcudf_exception_handler

    cdef unique_ptr[scalar] make_empty_scalar_like(
        const column_view &
    ) except +libcudf_exception_handler
