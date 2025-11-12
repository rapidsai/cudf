# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.types cimport mask_state, size_type

from pylibcudf.gpumemoryview cimport gpumemoryview

from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column


cpdef DeviceBuffer copy_bitmask(Column col, Stream stream=*, DeviceMemoryResource mr=*)

cpdef size_t bitmask_allocation_size_bytes(size_type number_of_bits)

cpdef DeviceBuffer create_null_mask(
    size_type size,
    mask_state state=*,
    Stream stream=*,
    DeviceMemoryResource mr=*
)

cpdef tuple bitmask_and(list columns, Stream stream=*, DeviceMemoryResource mr=*)

cpdef tuple bitmask_or(list columns, Stream stream=*, DeviceMemoryResource mr=*)

cpdef size_type null_count(
    gpumemoryview bitmask,
    size_type start,
    size_type stop,
    Stream stream=*
)
