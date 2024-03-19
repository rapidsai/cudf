# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.types cimport (
    MaskAndNullCnt,
    bitmask_type,
    mask_state,
    size_type,
)

from .column cimport Column
from .types cimport DataType, Id as TypeId

ctypedef fused MakeEmptyColumnOperand:
    DataType
    TypeId

ctypedef fused MaskArg:
    mask_state
    tuple
    object


cpdef Column make_empty_column(
    MakeEmptyColumnOperand type_or_id
)

cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    MaskArg mstate,
)
