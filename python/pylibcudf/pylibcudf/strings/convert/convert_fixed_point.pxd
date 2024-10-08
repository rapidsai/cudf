# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType


cpdef Column to_fixed_point(Column input, DataType output_type)

cpdef Column from_fixed_point(Column input)

cpdef Column is_fixed_point(Column input, DataType decimal_type=*)
