# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType


cpdef Column get_variant_field(
    Column variant_column,
    str path,
    object stream=*,
    object mr=*,
)


cpdef Column cast_variant(
    Column variant_column,
    DataType desired_type,
    object stream=*,
    object mr=*,
)


cpdef Column extract_variant_field(
    Column variant_column,
    str path,
    DataType desired_type,
    object stream=*,
    object mr=*,
)
