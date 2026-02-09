# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column edit_distance(
    Column input,
    Column targets,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column edit_distance_matrix(
    Column input,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
