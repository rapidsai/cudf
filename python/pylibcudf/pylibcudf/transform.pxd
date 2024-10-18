# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from pylibcudf.libcudf.types cimport bitmask_type, data_type

from .column cimport Column
from .expressions cimport Expression
from .gpumemoryview cimport gpumemoryview
from .table cimport Table
from .types cimport DataType


cpdef tuple[gpumemoryview, int] nans_to_nulls(Column input)

cpdef Column compute_column(Table input, Expression expr)

cpdef tuple[gpumemoryview, int] bools_to_mask(Column input)

cpdef Column mask_to_bools(Py_ssize_t bitmask, int begin_bit, int end_bit)

cpdef Column transform(Column input, str unary_udf, DataType output_type, bool is_ptx)

cpdef tuple[Table, Column] encode(Table input)

cpdef Table one_hot_encode(Column input_column, Column categories)
