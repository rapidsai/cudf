# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/utilities/traits.hpp" namespace "cudf" nogil:
    cdef bool is_relationally_comparable(data_type)
    cdef bool is_equality_comparable(data_type)
    cdef bool is_numeric(data_type)
    cdef bool is_numeric_not_bool(data_type)
    cdef bool is_index_type(data_type)
    cdef bool is_unsigned(data_type)
    cdef bool is_integral(data_type)
    cdef bool is_integral_not_bool(data_type)
    cdef bool is_floating_point(data_type)
    cdef bool is_boolean(data_type)
    cdef bool is_timestamp(data_type)
    cdef bool is_fixed_point(data_type)
    cdef bool is_duration(data_type)
    cdef bool is_chrono(data_type)
    cdef bool is_dictionary(data_type)
    cdef bool is_fixed_width(data_type)
    cdef bool is_compound(data_type)
    cdef bool is_nested(data_type)
    cdef bool is_bit_castable(data_type, data_type)
