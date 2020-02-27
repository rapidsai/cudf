# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import IntEnum

import numpy as np

from cudf._libxx.types cimport (
    underlying_type_t_order,
    underlying_type_t_null_order,
    underlying_type_t_sorted,
    underlying_type_t_interpolation
)
cimport cudf._libxx.cpp.types as cudf_types


np_to_cudf_types = {
    np.dtype("int8"): cudf_types.INT8,
    np.dtype("int16"): cudf_types.INT16,
    np.dtype("int32"): cudf_types.INT32,
    np.dtype("int64"): cudf_types.INT64,
    np.dtype("float32"): cudf_types.FLOAT32,
    np.dtype("float64"): cudf_types.FLOAT64,
    np.dtype("datetime64[s]"): cudf_types.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): cudf_types.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): cudf_types.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): cudf_types.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): cudf_types.STRING,
    np.dtype("bool"): cudf_types.BOOL8,
}

cudf_to_np_types = {
    cudf_types.INT8: np.dtype("int8"),
    cudf_types.INT16: np.dtype("int16"),
    cudf_types.INT32: np.dtype("int32"),
    cudf_types.INT64: np.dtype("int64"),
    cudf_types.FLOAT32: np.dtype("float32"),
    cudf_types.FLOAT64: np.dtype("float64"),
    cudf_types.TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    cudf_types.TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    cudf_types.TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    cudf_types.TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    cudf_types.STRING: np.dtype("object"),
    cudf_types.BOOL8: np.dtype("bool"),
}

class Interpolation(IntEnum):
    LINEAR = <underlying_type_t_interpolation> cudf_types.interpolation.LINEAR
    LOWER = <underlying_type_t_interpolation> cudf_types.interpolation.LOWER
    HIGHER = <underlying_type_t_interpolation> cudf_types.interpolation.HIGHER
    MIDPOINT = <underlying_type_t_interpolation> cudf_types.interpolation.MIDPOINT
    NEAREST = <underlying_type_t_interpolation> cudf_types.interpolation.NEAREST


class Order(IntEnum):
    ASCENDING = <underlying_type_t_order> cudf_types.order.ASCENDING
    DESCENDING = <underlying_type_t_order> cudf_types.order.DESCENDING


class Sorted(IntEnum):
    YES = <underlying_type_t_sorted> cudf_types.sorted.YES
    NO = <underlying_type_t_sorted> cudf_types.sorted.NO


class NullOrder(IntEnum):
    BEFORE = <underlying_type_t_order> cudf_types.null_order.BEFORE
    AFTER = <underlying_type_t_order> cudf_types.null_order.AFTER
