# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libcpp cimport bool as cbool
from pylibcudf.libcudf.copying cimport (
    mask_allocation_policy,
    out_of_bounds_policy,
)
from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrTable:
    Table
    Column


ctypedef fused TableOrListOfScalars:
    Table
    # The contents of the list must be validated as Scalars at runtime.
    list


# Need two separate fused types to generate the cartesian product of signatures.
ctypedef fused LeftCopyIfElseOperand:
    Column
    Scalar

ctypedef fused RightCopyIfElseOperand:
    Column
    Scalar


cpdef Table gather(
    Table source_table,
    Column gather_map,
    out_of_bounds_policy bounds_policy,
    Stream stream=*
)

cpdef Table scatter(
    TableOrListOfScalars source, Column scatter_map, Table target_table, Stream stream=*
)

cpdef ColumnOrTable empty_like(ColumnOrTable input)

cpdef Column allocate_like(
    Column input_column, mask_allocation_policy policy, size=*, Stream stream=*
)

cpdef Column copy_range_in_place(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
    Stream stream=*
)

cpdef Column copy_range(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
    Stream stream=*
)

cpdef Column shift(Column input, size_type offset, Scalar fill_value, Stream stream=*)

cpdef list slice(ColumnOrTable input, list indices, Stream stream=*)

cpdef list split(ColumnOrTable input, list splits, Stream stream=*)

cpdef Column copy_if_else(
    LeftCopyIfElseOperand lhs,
    RightCopyIfElseOperand rhs,
    Column boolean_mask,
    Stream stream=*
)

cpdef Table boolean_mask_scatter(
    TableOrListOfScalars input,
    Table target,
    Column boolean_mask,
    Stream stream=*
)

cpdef Scalar get_element(Column input_column, size_type index, Stream stream=*)
