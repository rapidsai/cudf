# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.stream_compaction cimport (
    duplicate_keep_option,
)
from cudf._lib.pylibcudf.libcudf.types cimport (
    nan_equality,
    nan_policy,
    null_equality,
    null_policy,
    size_type,
)

from .column cimport Column
from .table cimport Table


cpdef Table drop_nulls(Table source_table, list keys, size_type keep_threshold)

cpdef Table drop_nans(Table source_table, list keys, size_type keep_threshold)

cpdef Table unique(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
)

cpdef Table distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
)

cpdef Column distinct_indices(
    Table input,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
)

cpdef Table stable_distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
)

cpdef size_type unique_count(
    Column column,
    null_policy null_handling,
    nan_policy nan_handling
)

cpdef size_type distinct_count(
    Column column,
    null_policy null_handling,
    nan_policy nan_handling
)
