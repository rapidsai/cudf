# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.table cimport Table

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Table from_dlpack(
    object managed_tensor, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef object to_dlpack(Table input, Stream stream=*, DeviceMemoryResource mr=*)
