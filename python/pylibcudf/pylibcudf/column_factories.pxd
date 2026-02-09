# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.libcudf.types cimport mask_state
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .types cimport DataType, size_type, type_id

ctypedef fused MakeEmptyColumnOperand:
    DataType
    type_id
    object

ctypedef fused MaskArg:
    mask_state
    object


cpdef Column make_empty_column(
    MakeEmptyColumnOperand type_or_id,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    MaskArg mask,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column make_fixed_point_column(
    DataType type_,
    size_type size,
    MaskArg mask,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column make_timestamp_column(
    DataType type_,
    size_type size,
    MaskArg mask,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column make_duration_column(
    DataType type_,
    size_type size,
    MaskArg mask,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column make_fixed_width_column(
    DataType type_,
    size_type size,
    MaskArg mask,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)
