# Copyright (c) 2024, NVIDIA CORPORATION.


from cudf._lib.cpp.scalar.scalar cimport scalar
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

cdef extern from "cudf/scalar/scalar_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] make_string_scalar(const string & _string) except +