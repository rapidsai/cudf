# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import IntEnum

import numpy as np

from libcpp.memory cimport shared_ptr, make_shared

from cudf._lib.types cimport (
    underlying_type_t_order,
    underlying_type_t_null_order,
    underlying_type_t_sorted,
    underlying_type_t_interpolation
)
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf.core.dtypes import ListDtype, StructDtype

cimport cudf._lib.cpp.types as libcudf_types


class TypeId(IntEnum):
    EMPTY = <underlying_type_t_type_id> libcudf_types.type_id.EMPTY
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
    TIMESTAMP_DAYS = (
        <underlying_type_t_type_id> libcudf_types.type_id.TIMESTAMP_DAYS
    )
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
    DURATION_SECONDS = (
        <underlying_type_t_type_id> libcudf_types.type_id.DURATION_SECONDS
    )
    DURATION_MILLISECONDS = (
        <underlying_type_t_type_id> libcudf_types.type_id.DURATION_MILLISECONDS
    )
    DURATION_MICROSECONDS = (
        <underlying_type_t_type_id> libcudf_types.type_id.DURATION_MICROSECONDS
    )
    DURATION_NANOSECONDS = (
        <underlying_type_t_type_id> libcudf_types.type_id.DURATION_NANOSECONDS
    )


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
    np.dtype("timedelta64[s]"): TypeId.DURATION_SECONDS,
    np.dtype("timedelta64[ms]"): TypeId.DURATION_MILLISECONDS,
    np.dtype("timedelta64[us]"): TypeId.DURATION_MICROSECONDS,
    np.dtype("timedelta64[ns]"): TypeId.DURATION_NANOSECONDS,
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
    TypeId.DURATION_SECONDS: np.dtype("timedelta64[s]"),
    TypeId.DURATION_MILLISECONDS: np.dtype("timedelta64[ms]"),
    TypeId.DURATION_MICROSECONDS: np.dtype("timedelta64[us]"),
    TypeId.DURATION_NANOSECONDS: np.dtype("timedelta64[ns]"),
}

duration_unit_map = {
    TypeId.DURATION_SECONDS: "s",
    TypeId.DURATION_MILLISECONDS: "ms",
    TypeId.DURATION_MICROSECONDS: "us",
    TypeId.DURATION_NANOSECONDS: "ns"
}

datetime_unit_map = {
    TypeId.TIMESTAMP_SECONDS: "s",
    TypeId.TIMESTAMP_MILLISECONDS: "ms",
    TypeId.TIMESTAMP_MICROSECONDS: "us",
    TypeId.TIMESTAMP_NANOSECONDS: "ns",
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


cdef dtype_from_lists_column_view(column_view cv):
    # lists_column_view have no default constructor, so we heap
    # allocate it to get around Cython's limitation of requiring
    # default constructors for stack allocated objects
    cdef shared_ptr[lists_column_view] lv = make_shared[lists_column_view](cv)
    cdef column_view child = lv.get()[0].child()

    if child.type().id() == libcudf_types.type_id.LIST:
        return ListDtype(dtype_from_lists_column_view(child))
    elif child.type().id() == libcudf_types.type_id.EMPTY:
        return ListDtype(np.dtype("int8"))
    else:
        return ListDtype(
            dtype_from_column_view(child)
        )

cdef dtype_from_structs_column_view(column_view cv):
    fields = {
        str(i): dtype_from_column_view(cv.child(i))
        for i in range(cv.num_children())
    }
    return StructDtype(fields)

cdef dtype_from_column_view(column_view cv):
    cdef libcudf_types.type_id tid = cv.type().id()
    if tid == libcudf_types.type_id.LIST:
        dtype = dtype_from_lists_column_view(cv)
    elif tid == libcudf_types.type_id.STRUCT:
        dtype = dtype_from_structs_column_view(cv)
    else:
        dtype = cudf_to_np_types[<underlying_type_t_type_id>(tid)]
    return dtype
