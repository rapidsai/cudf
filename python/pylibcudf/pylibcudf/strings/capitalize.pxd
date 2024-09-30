# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar


cpdef Column capitalize(Column input, Scalar delimiters=*)
cpdef Column title(Column input)
cpdef Column is_title(Column input)
