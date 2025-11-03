# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.combine cimport (
    output_if_empty_list,
    separator_on_nulls,
)
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column concatenate(
    Table strings_columns,
    ColumnOrScalar separator,
    Scalar narep=*,
    Scalar col_narep=*,
    separator_on_nulls separate_nulls=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column join_strings(
    Column input,
    Scalar separator,
    Scalar narep,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column join_list_elements(
    Column source_strings,
    ColumnOrScalar separator,
    Scalar separator_narep,
    Scalar string_narep,
    separator_on_nulls separate_nulls,
    output_if_empty_list empty_list_policy,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
