# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type

cpdef Column build_suffix_array(Column input, size_type min_width)
cpdef Column resolve_duplicates(Column input, Column indices, size_type min_width)
cpdef Column resolve_duplicates_pair(
  Column input1, Column indices1, Column input2, Column indices2, size_type min_width
)
