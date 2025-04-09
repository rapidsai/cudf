# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.table cimport Table

from pylibcudf.libcudf.interop cimport column_metadata

cpdef Table from_dlpack(object managed_tensor)

cpdef object to_dlpack(Table input)
