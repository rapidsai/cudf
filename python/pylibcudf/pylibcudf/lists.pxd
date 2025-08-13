# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.libcudf.types cimport (
    nan_equality, null_equality, null_order, order, size_type
)
from pylibcudf.libcudf.copying cimport out_of_bounds_policy
from pylibcudf.libcudf.lists.combine cimport concatenate_null_policy
from pylibcudf.libcudf.lists.contains cimport duplicate_find_option
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrScalar:
    Column
    Scalar

ctypedef fused ColumnOrSizeType:
    Column
    size_type

cpdef Table explode_outer(Table, size_type explode_column_idx, Stream stream=*)

cpdef Column concatenate_rows(Table, Stream stream=*)

cpdef Column concatenate_list_elements(
    Column, concatenate_null_policy null_policy, Stream stream=*
)

cpdef Column contains(Column, ColumnOrScalar, Stream stream=*)

cpdef Column contains_nulls(Column, Stream stream=*)

cpdef Column index_of(
    Column, ColumnOrScalar, duplicate_find_option, Stream stream=*
)

cpdef Column reverse(Column, Stream stream=*)

cpdef Column segmented_gather(
    Column, Column, out_of_bounds_policy bounds_policy=*, Stream stream=*
)

cpdef Column extract_list_element(Column, ColumnOrSizeType, Stream stream=*)

cpdef Column count_elements(Column, Stream stream=*)

cpdef Column sequences(
    Column, Column, Column steps = *, Stream stream=*
)

cpdef Column sort_lists(
    Column, order, null_order, bool stable = *, Stream stream=*
)

cpdef Column difference_distinct(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    Stream stream=*
)

cpdef Column have_overlap(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    Stream stream=*
)

cpdef Column intersect_distinct(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    Stream stream=*
)

cpdef Column union_distinct(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    Stream stream=*
)

cpdef Column apply_boolean_mask(Column, Column, Stream stream=*)

cpdef Column distinct(Column, null_equality, nan_equality, Stream stream=*)
