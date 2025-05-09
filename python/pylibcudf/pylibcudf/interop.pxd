# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.table cimport Table

cpdef Table from_dlpack(object managed_tensor)

cpdef object to_dlpack(Table input)
cpdef object to_dlpack_col(
    Column col,
    stream=*,
    max_version=*,
    dl_device=*,
    copy=*)
