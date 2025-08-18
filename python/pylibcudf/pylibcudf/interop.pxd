# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream

cpdef Table from_dlpack(object managed_tensor, Stream stream=*)

cpdef object to_dlpack(Table input, Stream stream=*)
