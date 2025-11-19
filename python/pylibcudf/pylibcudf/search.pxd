# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table


cpdef Column lower_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column upper_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column contains(
    Column haystack, Column needles, Stream stream = *, DeviceMemoryResource mr = *
)
