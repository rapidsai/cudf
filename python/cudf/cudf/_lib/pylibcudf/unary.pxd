# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.unary cimport unary_operator

from .column cimport Column
from .types cimport DataType


cpdef Column unary_operation(Column input, unary_operator op)

cpdef Column is_null(Column input)

cpdef Column is_valid(Column input)

cpdef Column cast(Column input, DataType data_type)

cpdef Column is_nan(Column input)

cpdef Column is_not_nan(Column input)
