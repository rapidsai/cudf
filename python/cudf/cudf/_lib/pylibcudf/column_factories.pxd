# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.types cimport mask_state, size_type, type_id as TypeId

from .column cimport Column
from .types cimport DataType

ctypedef fused MakeEmptyColumnOperand:
    DataType
    TypeId

cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    mask_state state
)
