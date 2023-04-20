# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport device_buffer

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view, mutable_column_view
from cudf._lib.cpp.libcpp.functional cimport reference_wrapper
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.exception_handler cimport cudf_exception_handler

ctypedef const scalar constscalar

cdef extern from "cudf/contiguous_split.hpp" namespace "cudf::packed_columns" nogil:
    cdef struct metadata:
        metadata(vector[uint8_t]&& v)
        const uint8_t* data () except +
        size_type size () except +

cdef extern from "cudf/contiguous_split.hpp" namespace "cudf" nogil:
    cdef cppclass packed_columns:
        unique_ptr[metadata] metadata_
        unique_ptr[device_buffer] gpu_data

    cdef struct contiguous_split_result:
        table_view table
        vector[device_buffer] all_data

    cdef vector[contiguous_split_result] contiguous_split (
        table_view input_table,
        vector[size_type] splits
    ) except +

    cdef packed_columns pack (const table_view& input) except +

    cdef table_view unpack (const packed_columns& input) except +
