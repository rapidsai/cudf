# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type


cpdef Column jaccard_index(Column input1, Column input2, size_type width)
