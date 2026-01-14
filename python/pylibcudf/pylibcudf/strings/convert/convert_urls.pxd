# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column url_encode(
    Column Input, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column url_decode(
    Column Input, Stream stream=*, DeviceMemoryResource mr=*
)
