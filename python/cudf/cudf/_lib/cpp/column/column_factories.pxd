# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type, mask_state, size_type


cdef extern from "cudf/column/column_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] make_numeric_column(data_type type,
                                                size_type size,
                                                mask_state state) except +

    cdef unique_ptr[column] make_column_from_scalar (const scalar & s,
                                                     size_type size) except +
