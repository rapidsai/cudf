# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport (
    size_type,
)

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport scalar
from libcpp.memory cimport unique_ptr

cdef extern from "cudf/column/column_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] make_column_from_scalar (const scalar & s,
                                                     size_type size) except +
