# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.scalar cimport Scalar


cpdef Column capitalize(Column input, Scalar delimiters=*)
cpdef Column title(Column input)
cpdef Column is_title(Column input)
