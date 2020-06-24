# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import IntEnum

import numpy as np

from cudf._lib.types cimport (
    underlying_type_t_order,
    underlying_type_t_null_order,
    underlying_type_t_sorted,
    underlying_type_t_interpolation
)
cimport cudf._lib.cpp.types as libcudf_types


class TypeId(IntEnum):
    INT8 = <underlying_type_t_type_id> libcudf_types.type_id.INT8
    INT16 = <underlying_type_t_type_id> libcudf_types.type_id.INT16
    INT32 = <underlying_type_t_type_id> libcudf_types.type_id.INT32
    INT64 = <underlying_type_t_type_id> libcudf_types.type_id.INT64
    UINT8 = <underlying_type_t_type_id> libcudf_types.type_id.UINT8
    UINT16 = <underlying_type_t_type_id> libcudf_types.type_id.UINT16
    UINT32 = <underlying_type_t_type_id> libcudf_types.type_id.UINT32
    UINT64 = <underlying_type_t_type_id> libcudf_types.type_id.UINT64
    FLOAT32 = <underlying_type_t_type_id> libcudf_types.type_id.FLOAT32
    FLOAT64 = <underlying_type_t_type_id> libcudf_types.type_id.FLOAT64
    TIMESTAMP_SECONDS = (
        <underlying_type_t_type_id> libcudf_types.type_id.TIMESTAMP_SECONDS
    )
    TIMESTAMP_MILLISECONDS = (
        <underlying_type_t_type_id> (
            libcudf_types.type_id.TIMESTAMP_MILLISECONDS
        )
    )
    TIMESTAMP_MICROSECONDS = (
        <underlying_type_t_type_id> (
            libcudf_types.type_id.TIMESTAMP_MICROSECONDS
        )
    )
    TIMESTAMP_NANOSECONDS = (
        <underlying_type_t_type_id> libcudf_types.type_id.TIMESTAMP_NANOSECONDS
    )
    STRING = <underlying_type_t_type_id> libcudf_types.type_id.STRING
    BOOL8 = <underlying_type_t_type_id> libcudf_types.type_id.BOOL8


np_to_cudf_types = {
    np.dtype("int8"): TypeId.INT8,
    np.dtype("int16"): TypeId.INT16,
    np.dtype("int32"): TypeId.INT32,
    np.dtype("int64"): TypeId.INT64,
    np.dtype("uint8"): TypeId.UINT8,
    np.dtype("uint16"): TypeId.UINT16,
    np.dtype("uint32"): TypeId.UINT32,
    np.dtype("uint64"): TypeId.UINT64,
    np.dtype("float32"): TypeId.FLOAT32,
    np.dtype("float64"): TypeId.FLOAT64,
    np.dtype("datetime64[s]"): TypeId.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): TypeId.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): TypeId.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): TypeId.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): TypeId.STRING,
    np.dtype("bool"): TypeId.BOOL8,
}

cudf_to_np_types = {
    TypeId.INT8: np.dtype("int8"),
    TypeId.INT16: np.dtype("int16"),
    TypeId.INT32: np.dtype("int32"),
    TypeId.INT64: np.dtype("int64"),
    TypeId.UINT8: np.dtype("uint8"),
    TypeId.UINT16: np.dtype("uint16"),
    TypeId.UINT32: np.dtype("uint32"),
    TypeId.UINT64: np.dtype("uint64"),
    TypeId.FLOAT32: np.dtype("float32"),
    TypeId.FLOAT64: np.dtype("float64"),
    TypeId.TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    TypeId.TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    TypeId.TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    TypeId.TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    TypeId.STRING: np.dtype("object"),
    TypeId.BOOL8: np.dtype("bool"),
}


class Interpolation(IntEnum):
    LINEAR = (
        <underlying_type_t_interpolation> libcudf_types.interpolation.LINEAR
    )
    LOWER = (
        <underlying_type_t_interpolation> libcudf_types.interpolation.LOWER
    )
    HIGHER = (
        <underlying_type_t_interpolation> libcudf_types.interpolation.HIGHER
    )
    MIDPOINT = (
        <underlying_type_t_interpolation> libcudf_types.interpolation.MIDPOINT
    )
    NEAREST = (
        <underlying_type_t_interpolation> libcudf_types.interpolation.NEAREST
    )


class Order(IntEnum):
    ASCENDING = <underlying_type_t_order> libcudf_types.order.ASCENDING
    DESCENDING = <underlying_type_t_order> libcudf_types.order.DESCENDING


class Sorted(IntEnum):
    YES = <underlying_type_t_sorted> libcudf_types.sorted.YES
    NO = <underlying_type_t_sorted> libcudf_types.sorted.NO


class NullOrder(IntEnum):
    BEFORE = <underlying_type_t_order> libcudf_types.null_order.BEFORE
    AFTER = <underlying_type_t_order> libcudf_types.null_order.AFTER


class NullHandling(IntEnum):
    INCLUDE = <underlying_type_t_null_policy> libcudf_types.null_policy.INCLUDE
    EXCLUDE = <underlying_type_t_null_policy> libcudf_types.null_policy.EXCLUDE
