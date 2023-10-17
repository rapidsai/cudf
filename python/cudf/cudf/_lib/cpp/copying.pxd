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

cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:
    cpdef enum class out_of_bounds_policy(bool):
        NULLIFY
        DONT_CHECK

    cdef unique_ptr[table] gather (
        const table_view& source_table,
        const column_view& gather_map,
        out_of_bounds_policy policy
    ) except +cudf_exception_handler

    cdef unique_ptr[column] shift(
        const column_view& input,
        size_type offset,
        const scalar& fill_values
    ) except +

    cdef unique_ptr[table] scatter (
        table_view source_table,
        column_view scatter_map,
        table_view target_table,
    ) except +

    cdef unique_ptr[table] scatter (
        vector[reference_wrapper[constscalar]] source_scalars,
        column_view indices,
        table_view target,
    ) except +

    ctypedef enum mask_allocation_policy:
        NEVER 'cudf::mask_allocation_policy::NEVER',
        RETAIN 'cudf::mask_allocation_policy::RETAIN',
        ALWAYS 'cudf::mask_allocation_policy::ALWAYS'

    cdef unique_ptr[column] empty_like (
        column_view input_column
    ) except +

    cdef unique_ptr[column] allocate_like (
        column_view input_column,
        mask_allocation_policy policy
    ) except +

    cdef unique_ptr[column] allocate_like (
        column_view input_column,
        size_type size,
        mask_allocation_policy policy
    ) except +

    cdef unique_ptr[table] empty_like (
        table_view input_table
    ) except +

    cdef void copy_range_in_place (
        column_view input_column,
        mutable_column_view target_column,
        size_type input_begin,
        size_type input_end,
        size_type target_begin
    ) except +

    cdef unique_ptr[column] copy_range (
        column_view input_column,
        column_view target_column,
        size_type input_begin,
        size_type input_end,
        size_type target_begin
    ) except +

    cdef vector[column_view] slice (
        column_view input_column,
        vector[size_type] indices
    ) except +

    cdef vector[table_view] slice (
        table_view input_table,
        vector[size_type] indices
    ) except +

    cdef vector[column_view] split (
        column_view input_column,
        vector[size_type] splits
    ) except +

    cdef vector[table_view] split (
        table_view input_table,
        vector[size_type] splits
    ) except +

    cdef unique_ptr[column] copy_if_else (
        column_view lhs,
        column_view rhs,
        column_view boolean_mask
    ) except +

    cdef unique_ptr[column] copy_if_else (
        scalar lhs,
        column_view rhs,
        column_view boolean_mask
    ) except +

    cdef unique_ptr[column] copy_if_else (
        column_view lhs,
        scalar rhs,
        column_view boolean_mask
    ) except +

    cdef unique_ptr[column] copy_if_else (
        scalar lhs,
        scalar rhs,
        column_view boolean_mask
    ) except +

    cdef unique_ptr[table] boolean_mask_scatter (
        table_view input,
        table_view target,
        column_view boolean_mask
    ) except +

    cdef unique_ptr[table] boolean_mask_scatter (
        vector[reference_wrapper[constscalar]] input,
        table_view target,
        column_view boolean_mask
    ) except +

    cdef unique_ptr[scalar] get_element (
        column_view input,
        size_type index
    ) except +

    ctypedef enum sample_with_replacement:
        FALSE 'cudf::sample_with_replacement::FALSE',
        TRUE 'cudf::sample_with_replacement::TRUE',
