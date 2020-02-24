# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *
from enum import Enum


np_to_cudf_types = {
    np.dtype("int8"): INT8,
    np.dtype("int16"): INT16,
    np.dtype("int32"): INT32,
    np.dtype("int64"): INT64,
    np.dtype("float32"): FLOAT32,
    np.dtype("float64"): FLOAT64,
    np.dtype("datetime64[s]"): TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): TIMESTAMP_NANOSECONDS,
    np.dtype("object"): STRING,
    np.dtype("bool"): BOOL8,
}

cudf_to_np_types = {
    INT8: np.dtype("int8"),
    INT16: np.dtype("int16"),
    INT32: np.dtype("int32"),
    INT64: np.dtype("int64"),
    FLOAT32: np.dtype("float32"),
    FLOAT64: np.dtype("float64"),
    TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    STRING: np.dtype("object"),
    BOOL8: np.dtype("bool"),
}

class MaskState(Enum):
    """
    Enum for null mask creation state
    """
    UNALLOCATED   = <underlying_type_t_mask_state> mask_state.UNALLOCATED
    UNINITIALIZED = <underlying_type_t_mask_state> mask_state.UNINITIALIZED
    ALL_VALID     = <underlying_type_t_mask_state> mask_state.ALL_VALID
    ALL_NULL      = <underlying_type_t_mask_state> mask_state.ALL_NULL

class Interpolation(Enum):
    LINEAR   = <underlying_type_t_interpolation> interpolation.LINEAR
    LOWER    = <underlying_type_t_interpolation> interpolation.LOWER
    HIGHER   = <underlying_type_t_interpolation> interpolation.HIGHER
    MIDPOINT = <underlying_type_t_interpolation> interpolation.MIDPOINT
    NEAREST  = <underlying_type_t_interpolation> interpolation.NEAREST

class Order(Enum):
    ASCENDING  = <underlying_type_t_order> order.ASCENDING
    DESCENDING = <underlying_type_t_order> order.DESCENDING

class Sorted(Enum):
    YES = <underlying_type_t_sorted> sorted.YES
    NO  = <underlying_type_t_sorted> sorted.NO

class NullOrder(Enum):
    BEFORE = <underlying_type_t_order> null_order.BEFORE
    AFTER  = <underlying_type_t_order> null_order.AFTER
