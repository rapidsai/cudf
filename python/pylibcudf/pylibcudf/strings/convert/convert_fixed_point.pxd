# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column to_fixed_point(
    Column input,
    DataType output_type,
    Stream stream=*,
    DeviceMemoryResource mr=*
)

cpdef Column from_fixed_point(
    Column input, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column is_fixed_point(
    Column input,
    DataType decimal_type=*,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
