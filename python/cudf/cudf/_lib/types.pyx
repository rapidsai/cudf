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
from cudf.core.dtypes import ListDtype

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.cpp.types cimport data_type
from cudf.core.dtypes import (
    Int8Dtype,
    Int16Dtype,
    Int32Dtype, 
    Int64Dtype, 
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
    Float32Dtype,
    Float64Dtype,
    StringDtype,
    BooleanDtype,
    Timedelta64NSDtype,
    Timedelta64USDtype,
    Timedelta64MSDtype,
    Timedelta64SDtype,
    Datetime64NSDtype,
    Datetime64USDtype,
    Datetime64MSDtype,
    Datetime64SDtype,
)


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
    Int8Dtype(): TypeId.INT8,
    Int16Dtype(): TypeId.INT16,
    Int32Dtype(): TypeId.INT32,
    Int64Dtype(): TypeId.INT64,
    UInt8Dtype(): TypeId.UINT8,
    UInt16Dtype(): TypeId.UINT16,
    UInt32Dtype(): TypeId.UINT32,
    UInt64Dtype(): TypeId.UINT64,
    Float32Dtype(): TypeId.FLOAT32,
    Float64Dtype(): TypeId.FLOAT64,
    Datetime64SDtype(): TypeId.TIMESTAMP_SECONDS,
    Datetime64MSDtype(): TypeId.TIMESTAMP_MILLISECONDS,
    Datetime64USDtype(): TypeId.TIMESTAMP_MICROSECONDS,
    Datetime64NSDtype(): TypeId.TIMESTAMP_NANOSECONDS,
    StringDtype(): TypeId.STRING,
    BooleanDtype(): TypeId.BOOL8,
    Timedelta64SDtype(): TypeId.DURATION_SECONDS,
    Timedelta64MSDtype(): TypeId.DURATION_MILLISECONDS,
    Timedelta64USDtype(): TypeId.DURATION_MICROSECONDS,
    Timedelta64NSDtype(): TypeId.DURATION_NANOSECONDS,
}

cudf_to_np_types = {
    TypeId.INT8: Int8Dtype(),
    TypeId.INT16: Int16Dtype(),
    TypeId.INT32: Int32Dtype(),
    TypeId.INT64: Int64Dtype(),
    TypeId.UINT8: UInt8Dtype(),
    TypeId.UINT16: UInt16Dtype(),
    TypeId.UINT32: UInt32Dtype(),
    TypeId.UINT64: UInt64Dtype(),
    TypeId.FLOAT32: Float32Dtype(),
    TypeId.FLOAT64: Float64Dtype(),
    TypeId.TIMESTAMP_SECONDS: Datetime64SDtype(),
    TypeId.TIMESTAMP_MILLISECONDS: Datetime64MSDtype(),
    TypeId.TIMESTAMP_MICROSECONDS: Datetime64USDtype(),
    TypeId.TIMESTAMP_NANOSECONDS: Datetime64NSDtype(),
    TypeId.STRING: StringDtype(),
    TypeId.BOOL8: BooleanDtype(),
    TypeId.DURATION_SECONDS: Timedelta64SDtype(),
    TypeId.DURATION_MILLISECONDS: Timedelta64MSDtype(),
    TypeId.DURATION_MICROSECONDS: Timedelta64USDtype(),
    TypeId.DURATION_NANOSECONDS: Timedelta64NSDtype(),
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


cdef class _Dtype:
    cdef data_type get_libcudf_type(self) except *:

        cdef libcudf_types.type_id tid
        cdef data_type libcudf_type 

        if not isinstance(self, ListDtype):
            tid = <libcudf_types.type_id> (
                    <underlying_type_t_type_id> (
                        np_to_cudf_types[self]
                    )
                )
        else:
            tid = libcudf_types.type_id.LIST
        
        libcudf_type = libcudf_types.data_type(tid)
        return libcudf_type


cdef dtype_from_lists_column_view(column_view cv):
    # lists_column_view have no default constructor, so we heap
    # allocate it to get around Cython's limitation of requiring
    # default constructors for stack allocated objects
    cdef shared_ptr[lists_column_view] lv = make_shared[lists_column_view](cv)
    cdef column_view child = lv.get()[0].child()

    if child.type().id() == libcudf_types.type_id.LIST:
        return ListDtype(dtype_from_lists_column_view(child))
    else:
        return ListDtype(
            cudf_to_np_types[<underlying_type_t_type_id> child.type().id()]
        )


cdef dtype_from_column_view(column_view cv):
    cdef libcudf_types.type_id tid = cv.type().id()
    if tid == libcudf_types.type_id.LIST:
        dtype = dtype_from_lists_column_view(cv)
    else:
        dtype = cudf_to_np_types[<underlying_type_t_type_id>(tid)]
    return dtype
