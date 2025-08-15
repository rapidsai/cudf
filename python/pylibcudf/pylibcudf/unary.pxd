# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.libcudf.unary cimport unary_operator
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .types cimport DataType


cpdef Column unary_operation(Column input, unary_operator op, Stream stream = *)

cpdef Column is_null(Column input, Stream stream = *)

cpdef Column is_valid(Column input, Stream stream = *)

cpdef Column cast(Column input, DataType data_type, Stream stream = *)

cpdef Column is_nan(Column input, Stream stream = *)

cpdef Column is_not_nan(Column input, Stream stream = *)

cpdef bool is_supported_cast(DataType from_, DataType to)
