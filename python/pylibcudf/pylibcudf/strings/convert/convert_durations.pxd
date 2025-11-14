# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string
from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column to_durations(
    Column input,
    DataType duration_type,
    str format,
    Stream stream=*,
    DeviceMemoryResource mr=*
)

cpdef Column from_durations(
    Column durations,
    str format=*,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
