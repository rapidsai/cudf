# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.table cimport Table


cpdef Table from_dlpack(object managed_tensor)

cpdef object to_dlpack(Table input)
