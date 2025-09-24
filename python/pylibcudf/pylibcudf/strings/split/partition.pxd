# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream


cpdef Table partition(Column input, Scalar delimiter=*, Stream stream=*)

cpdef Table rpartition(Column input, Scalar delimiter=*, Stream stream=*)
