# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column count_characters(
    Column source_strings, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column count_bytes(
    Column source_strings, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column code_points(
    Column source_strings, Stream stream=*, DeviceMemoryResource mr=*
)
