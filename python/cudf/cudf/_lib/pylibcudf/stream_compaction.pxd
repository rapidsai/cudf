# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.cpp.stream_compaction cimport duplicate_keep_option
from cudf._lib.cpp.types cimport (
    nan_equality,
    nan_policy,
    null_equality,
    null_policy,
    size_type,
)

from .column cimport Column
from .table cimport Table


cpdef Table drop_nulls(Table source_table, list keys, size_type keep_threshold)

cpdef Table apply_boolean_mask(Table source_table, Column boolean_mask)

cpdef size_type distinct_count(
    Column source_table,
    null_policy null_handling,
    nan_policy nan_handling
)

cpdef Table stable_distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
)

cpdef Column distinct_indices(
    Table input,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
)
