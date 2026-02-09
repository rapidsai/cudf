# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.stream_compaction cimport duplicate_keep_option
from pylibcudf.libcudf.types cimport (
    nan_equality,
    nan_policy,
    null_equality,
    null_policy,
    size_type,
)
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table


cpdef Table drop_nulls(
    Table source_table,
    list keys,
    size_type keep_threshold,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table drop_nans(
    Table source_table,
    list keys,
    size_type keep_threshold,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table apply_boolean_mask(
    Table source_table,
    Column boolean_mask,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table unique(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column distinct_indices(
    Table input,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table stable_distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef size_type unique_count(
    Column column,
    null_policy null_handling,
    nan_policy nan_handling,
    Stream stream = *
)

cpdef size_type distinct_count(
    Column column,
    null_policy null_handling,
    nan_policy nan_handling,
    Stream stream = *
)

cpdef Table filter(
    Table predicate_table,
    Expression predicate_expr,
    Table filter_table,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)
