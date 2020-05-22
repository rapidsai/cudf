# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr

from rmm._lib.device_buffer cimport device_buffer

from cudf._lib.cpp.types cimport (
    size_type,
    data_type,
)
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cudf/transform.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[device_buffer], size_type] bools_to_mask (
        column_view input
    ) except +

    cdef pair[unique_ptr[device_buffer], size_type] nans_to_nulls(
        column_view input
    ) except +

    cdef unique_ptr[column] transform(
        column_view input,
        string unary_udf,
        data_type output_type,
        bool is_ptx
    ) except +
