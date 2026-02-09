# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column ipv4_to_integers(
    Column input, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column integers_to_ipv4(
    Column integers, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column is_ipv4(
    Column input, Stream stream=*, DeviceMemoryResource mr=*
)
