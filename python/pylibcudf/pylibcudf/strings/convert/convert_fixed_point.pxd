# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType
from rmm.pylibrmm.stream cimport Stream


cpdef Column to_fixed_point(Column input, DataType output_type, Stream stream=*)

cpdef Column from_fixed_point(Column input, Stream stream=*)

cpdef Column is_fixed_point(Column input, DataType decimal_type=*, Stream stream=*)
