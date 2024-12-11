# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table


cpdef Table partition(Column input, Scalar delimiter=*)

cpdef Table rpartition(Column input, Scalar delimiter=*)
