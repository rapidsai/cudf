# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.gpumemoryresource cimport DeviceMemoryResource
from pylibcudf.stream cimport Stream


cpdef Column decode_protobuf(
    Column binary_input,
    list schema,
    list default_ints,
    list default_floats,
    list default_bools,
    list default_strings,
    list enum_valid_values,
    list enum_names,
    bint fail_on_errors,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)
