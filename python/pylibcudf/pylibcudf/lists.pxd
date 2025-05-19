# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.libcudf.types cimport (
    nan_equality, null_equality, null_order, order, size_type
)
from pylibcudf.libcudf.lists.combine cimport concatenate_null_policy
from pylibcudf.libcudf.lists.contains cimport duplicate_find_option

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrScalar:
    Column
    Scalar

ctypedef fused ColumnOrSizeType:
    Column
    size_type

cpdef Table explode_outer(Table, size_type explode_column_idx)

cpdef Column concatenate_rows(Table)

cpdef Column concatenate_list_elements(Column, concatenate_null_policy null_policy)

cpdef Column contains(Column, ColumnOrScalar)

cpdef Column contains_nulls(Column)

cpdef Column index_of(Column, ColumnOrScalar, duplicate_find_option)

cpdef Column reverse(Column)

cpdef Column segmented_gather(Column, Column)

cpdef Column extract_list_element(Column, ColumnOrSizeType)

cpdef Column count_elements(Column)

cpdef Column sequences(Column, Column, Column steps = *)

cpdef Column sort_lists(Column, order, null_order, bool stable = *)

cpdef Column difference_distinct(
    Column, Column, null_equality nulls_equal=*, nan_equality nans_equal=*
)

cpdef Column have_overlap(
    Column, Column, null_equality nulls_equal=*, nan_equality nans_equal=*
)

cpdef Column intersect_distinct(
    Column, Column, null_equality nulls_equal=*, nan_equality nans_equal=*
)

cpdef Column union_distinct(
    Column, Column, null_equality nulls_equal=*, nan_equality nans_equal=*
)

cpdef Column apply_boolean_mask(Column, Column)

cpdef Column distinct(Column, null_equality, nan_equality)
