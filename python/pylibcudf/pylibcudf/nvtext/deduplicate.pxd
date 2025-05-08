# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type

cpdef Column substring_duplicates(Column input, size_type min_width)
cpdef Column build_suffix_array(Column input, size_type min_width)
