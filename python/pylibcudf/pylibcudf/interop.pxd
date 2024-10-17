# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.table cimport Table


cpdef Table from_dlpack(managed_tensor)

cpdef to_dlpack(Table input)
