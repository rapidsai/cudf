# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from rmm.pylibrmm.stream cimport Stream


cpdef Column to_lower(Column input, Stream stream=*)
cpdef Column to_upper(Column input, Stream stream=*)
cpdef Column swapcase(Column input, Stream stream=*)
