# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column to_lower(Column input, Stream stream=*, DeviceMemoryResource mr=*)
cpdef Column to_upper(Column input, Stream stream=*, DeviceMemoryResource mr=*)
cpdef Column swapcase(Column input, Stream stream=*, DeviceMemoryResource mr=*)
