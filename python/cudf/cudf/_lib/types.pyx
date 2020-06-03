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


np_to_cudf_types = {
    np.dtype("int8"): libcudf_types.INT8,
    np.dtype("int16"): libcudf_types.INT16,
    np.dtype("int32"): libcudf_types.INT32,
    np.dtype("int64"): libcudf_types.INT64,
    np.dtype("float32"): libcudf_types.FLOAT32,
    np.dtype("float64"): libcudf_types.FLOAT64,
    np.dtype("datetime64[s]"): libcudf_types.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): libcudf_types.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): libcudf_types.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): libcudf_types.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): libcudf_types.STRING,
    np.dtype("bool"): libcudf_types.BOOL8,
}

cudf_to_np_types = {
    libcudf_types.INT8: np.dtype("int8"),
    libcudf_types.INT16: np.dtype("int16"),
    libcudf_types.INT32: np.dtype("int32"),
    libcudf_types.INT64: np.dtype("int64"),
    libcudf_types.FLOAT32: np.dtype("float32"),
    libcudf_types.FLOAT64: np.dtype("float64"),
    libcudf_types.TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    libcudf_types.TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    libcudf_types.TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    libcudf_types.TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    libcudf_types.STRING: np.dtype("object"),
    libcudf_types.BOOL8: np.dtype("bool"),
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
