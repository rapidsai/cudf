# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp cimport bool as cbool

from cudf._lib.cpp.copying cimport mask_allocation_policy, out_of_bounds_policy
from cudf._lib.cpp.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table


cpdef Table gather(
    Table source_table,
    Column gather_map,
    out_of_bounds_policy bounds_policy
)

cpdef Table scatter_table(Table source, Column scatter_map, Table target_table)

cpdef Table scatter_scalars(list source, Column scatter_map, Table target_table)

cpdef object empty_column_like(Column input)

cpdef object empty_table_like(Table input)

cpdef Column allocate_like(Column input_column, mask_allocation_policy policy, size=*)

cpdef Column copy_range(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
)

cpdef Column shift(Column input, size_type offset, Scalar fill_values)

cpdef Column copy_if_else(object lhs, object rhs, Column boolean_mask)

cpdef Table boolean_mask_table_scatter(Table input, Table target, Column boolean_mask)

cpdef Table boolean_mask_scalars_scatter(list input, Table target, Column boolean_mask)
