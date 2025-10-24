# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Table partition(
    Column input, Scalar delimiter=*, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Table rpartition(
    Column input, Scalar delimiter=*, Stream stream=*, DeviceMemoryResource mr=*
)
