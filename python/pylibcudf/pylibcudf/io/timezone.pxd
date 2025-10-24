# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from ..table cimport Table
from .types cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Table make_timezone_transition_table(
    str tzif_dir, str timezone_name, Stream stream=*, DeviceMemoryResource mr=*
)
