# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType


cpdef Column to_floats(Column strings, DataType output_type)

cpdef Column from_floats(Column floats)

cpdef Column is_float(Column input)
