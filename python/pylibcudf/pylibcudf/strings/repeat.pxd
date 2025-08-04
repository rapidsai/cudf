# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.stream cimport Stream

ctypedef fused ColumnorSizeType:
    Column
    size_type

cpdef Column repeat_strings(
    Column input, ColumnorSizeType repeat_times, Stream stream=*
)
