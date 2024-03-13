# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.types cimport (
    MaskAndNullCnt,
    bitmask_type,
    mask_state,
    size_type,
    type_id as TypeId,
)

from .column cimport Column
from .types cimport DataType

ctypedef fused MakeEmptyColumnOperand:
    DataType
    TypeId

ctypedef fused MaskStateOrMask:
    mask_state
    MaskAndNullCnt

cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    MaskStateOrMask state_or_mask
)

cpdef Column make_empty_column(
    MakeEmptyColumnOperand type_or_id
)

cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    MaskStateOrMask state_or_mask
)

cdef Column make_timestamp_column(
    DataType type_,
    size_type size,
    MaskStateOrMask state_or_mask
)

cdef Column make_duration_column(
    DataType type_,
    size_type size,
    MaskStateOrMask state_or_mask
)

cdef Column make_fixed_point_column(
    DataType type_,
    size_type size,
    MaskStateOrMask state_or_mask
)
