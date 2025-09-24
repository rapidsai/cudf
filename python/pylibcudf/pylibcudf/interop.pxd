# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cpdef Table from_dlpack(
    object managed_tensor, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef object to_dlpack(Table input, Stream stream=*, DeviceMemoryResource mr=*)

cpdef _get_dlpack_device()

cpdef object to_dlpack_col(
    Column col,
    stream=*,
    max_version=*,
    dl_device=*,
    copy=*)
