# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column to_booleans(
    Column input, Scalar true_string, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column from_booleans(
    Column booleans,
    Scalar true_string,
    Scalar false_string,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
