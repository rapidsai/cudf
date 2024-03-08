# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type, mask_state, size_type, type_id


cdef extern from "cudf/column/column_factories.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] make_numeric_column(data_type type,
                                                size_type size,
                                                mask_state state) except +

    cdef unique_ptr[column] make_fixed_point_column(
        data_type type,
        size_type size,
        mask_state state) except +

    cdef unique_ptr[column] make_timestamp_column(
        data_type type,
        size_type size,
        mask_state state) except +

    cdef unique_ptr[column] make_duration_column(
        data_type type,
        size_type size,
        mask_state state) except +

    cdef unique_ptr[column] make_fixed_point_column(
        data_type type,
        size_type size,
        mask_state state) except +

    cdef unique_ptr[column] make_column_from_scalar (const scalar & s,
                                                     size_type size) except +

    cdef unique_ptr[column] make_dictionary_from_scalar(const scalar & s,
                                                        size_type size) except +

    cdef unique_ptr[column] make_empty_column(type_id id) except +
    cdef unique_ptr[column] make_empty_column(data_type type_) except +

    cdef unique_ptr[column] make_dictionary_column(
        unique_ptr[column] keys_column,
        unique_ptr[column] indices_column) except +
