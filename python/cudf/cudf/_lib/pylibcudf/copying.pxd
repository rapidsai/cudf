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

cpdef Column shift(Column input, size_type offset, Scalar fill_values)

cpdef Table scatter(object source, Column scatter_map, Table target_table)

cpdef object empty_like(object input)

cpdef Column allocate_like(Column input_column, mask_allocation_policy policy, size=*)
