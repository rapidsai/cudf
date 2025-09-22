# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from .table cimport Table

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Table transpose(Table input_table, Stream stream=*, DeviceMemoryResource mr=*)
