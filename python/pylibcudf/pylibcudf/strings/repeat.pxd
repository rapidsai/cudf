# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type

ctypedef fused ColumnorSizeType:
    Column
    size_type

cpdef Column repeat_strings(Column input, ColumnorSizeType repeat_times)
