# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType
from rmm.pylibrmm.stream cimport Stream


cpdef Column to_integers(Column input, DataType output_type, Stream stream=*)

cpdef Column from_integers(Column integers, Stream stream=*)

cpdef Column is_integer(Column input, DataType int_type=*, Stream stream=*)

cpdef Column hex_to_integers(Column input, DataType output_type, Stream stream=*)

cpdef Column is_hex(Column input, Stream stream=*)

cpdef Column integers_to_hex(Column input, Stream stream=*)
