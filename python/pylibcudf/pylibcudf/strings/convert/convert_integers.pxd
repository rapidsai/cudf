# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType


cpdef Column to_integers(Column input, DataType output_type)

cpdef Column from_integers(Column integers)

cpdef Column is_integer(Column input, DataType int_type=*)

cpdef Column hex_to_integers(Column input, DataType output_type)

cpdef Column is_hex(Column input)

cpdef Column integers_to_hex(Column input)
