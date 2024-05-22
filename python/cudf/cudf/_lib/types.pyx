# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from enum import IntEnum

import numpy as np
import pandas as pd

from libcpp.memory cimport make_shared, shared_ptr

cimport cudf._lib.pylibcudf.libcudf.types as libcudf_types
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.types cimport (
    underlying_type_t_interpolation,
    underlying_type_t_order,
    underlying_type_t_sorted,
)

import cudf
from cudf._lib import pylibcudf

size_type_dtype = np.dtype("int32")


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
    BOOL8 = <underlying_type_t_type_id> libcudf_types.type_id.BOOL8
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
    STRING = <underlying_type_t_type_id> libcudf_types.type_id.STRING
    DECIMAL32 = <underlying_type_t_type_id> libcudf_types.type_id.DECIMAL32
    DECIMAL64 = <underlying_type_t_type_id> libcudf_types.type_id.DECIMAL64
    DECIMAL128 = <underlying_type_t_type_id> libcudf_types.type_id.DECIMAL128
    STRUCT = <underlying_type_t_type_id> libcudf_types.type_id.STRUCT


SUPPORTED_NUMPY_TO_LIBCUDF_TYPES = {
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

SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES = {
    k: pylibcudf.TypeId(v).value
    for k, v in SUPPORTED_NUMPY_TO_LIBCUDF_TYPES.items()
}

LIBCUDF_TO_SUPPORTED_NUMPY_TYPES = {
    # There's no equivalent to EMPTY in cudf.  We translate EMPTY
    # columns from libcudf to ``int8`` columns of all nulls in Python.
    # ``int8`` is chosen because it uses the least amount of memory.
    TypeId.EMPTY: np.dtype("int8"),
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
    TypeId.BOOL8: np.dtype("bool"),
    TypeId.TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    TypeId.TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    TypeId.TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    TypeId.TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    TypeId.DURATION_SECONDS: np.dtype("timedelta64[s]"),
    TypeId.DURATION_MILLISECONDS: np.dtype("timedelta64[ms]"),
    TypeId.DURATION_MICROSECONDS: np.dtype("timedelta64[us]"),
    TypeId.DURATION_NANOSECONDS: np.dtype("timedelta64[ns]"),
    TypeId.STRING: np.dtype("object"),
    TypeId.STRUCT: np.dtype("object"),
}

PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES = {
    pylibcudf.TypeId(k).value: v
    for k, v in LIBCUDF_TO_SUPPORTED_NUMPY_TYPES.items()
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
        return cudf.ListDtype(dtype_from_lists_column_view(child))
    elif child.type().id() == libcudf_types.type_id.EMPTY:
        return cudf.ListDtype("int8")
    else:
        return cudf.ListDtype(
            dtype_from_column_view(child)
        )

cdef dtype_from_structs_column_view(column_view cv):
    fields = {
        str(i): dtype_from_column_view(cv.child(i))
        for i in range(cv.num_children())
    }
    return cudf.StructDtype(fields)

cdef dtype_from_column_view(column_view cv):
    cdef libcudf_types.type_id tid = cv.type().id()
    if tid == libcudf_types.type_id.LIST:
        return dtype_from_lists_column_view(cv)
    elif tid == libcudf_types.type_id.STRUCT:
        return dtype_from_structs_column_view(cv)
    elif tid == libcudf_types.type_id.DECIMAL64:
        return cudf.Decimal64Dtype(
            precision=cudf.Decimal64Dtype.MAX_PRECISION,
            scale=-cv.type().scale()
        )
    elif tid == libcudf_types.type_id.DECIMAL32:
        return cudf.Decimal32Dtype(
            precision=cudf.Decimal32Dtype.MAX_PRECISION,
            scale=-cv.type().scale()
        )
    elif tid == libcudf_types.type_id.DECIMAL128:
        return cudf.Decimal128Dtype(
            precision=cudf.Decimal128Dtype.MAX_PRECISION,
            scale=-cv.type().scale()
        )
    else:
        return LIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
            <underlying_type_t_type_id>(tid)
        ]

cdef libcudf_types.data_type dtype_to_data_type(dtype) except *:
    cdef libcudf_types.type_id tid
    if isinstance(dtype, cudf.ListDtype):
        tid = libcudf_types.type_id.LIST
    elif isinstance(dtype, cudf.StructDtype):
        tid = libcudf_types.type_id.STRUCT
    elif isinstance(dtype, cudf.Decimal128Dtype):
        tid = libcudf_types.type_id.DECIMAL128
    elif isinstance(dtype, cudf.Decimal64Dtype):
        tid = libcudf_types.type_id.DECIMAL64
    elif isinstance(dtype, cudf.Decimal32Dtype):
        tid = libcudf_types.type_id.DECIMAL32
    else:
        tid = <libcudf_types.type_id> (
            <underlying_type_t_type_id> (
                SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[np.dtype(dtype)]))

    if is_decimal_type_id(tid):
        return libcudf_types.data_type(tid, -dtype.scale)
    else:
        return libcudf_types.data_type(tid)

cpdef dtype_to_pylibcudf_type(dtype):
    if isinstance(dtype, cudf.ListDtype):
        return pylibcudf.DataType(pylibcudf.TypeId.LIST)
    elif isinstance(dtype, cudf.StructDtype):
        return pylibcudf.DataType(pylibcudf.TypeId.STRUCT)
    elif isinstance(dtype, cudf.Decimal128Dtype):
        tid = pylibcudf.TypeId.DECIMAL128
        return pylibcudf.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal64Dtype):
        tid = pylibcudf.TypeId.DECIMAL64
        return pylibcudf.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal32Dtype):
        tid = pylibcudf.TypeId.DECIMAL32
        return pylibcudf.DataType(tid, -dtype.scale)
    # libcudf types don't support localization so convert to the base type
    elif isinstance(dtype, pd.DatetimeTZDtype):
        dtype = np.dtype(f"<M8[{dtype.unit}]")
    else:
        dtype = np.dtype(dtype)
    return pylibcudf.DataType(SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES[dtype])

cdef bool is_decimal_type_id(libcudf_types.type_id tid) except *:
    return tid in (
        libcudf_types.type_id.DECIMAL128,
        libcudf_types.type_id.DECIMAL64,
        libcudf_types.type_id.DECIMAL32,
    )


def dtype_from_pylibcudf_lists_column(col):
    child = col.list_view().child()
    tid = child.type().id()

    if tid == pylibcudf.TypeId.LIST:
        return cudf.ListDtype(dtype_from_pylibcudf_lists_column(child))
    elif tid == pylibcudf.TypeId.EMPTY:
        return cudf.ListDtype("int8")
    else:
        return cudf.ListDtype(
            dtype_from_pylibcudf_column(child)
        )


def dtype_from_pylibcudf_structs_column(col):
    fields = {
        str(i): dtype_from_pylibcudf_column(col.child(i))
        for i in range(col.num_children())
    }
    return cudf.StructDtype(fields)


def dtype_from_pylibcudf_column(col):
    type_ = col.type()
    tid = type_.id()

    if tid == pylibcudf.TypeId.LIST:
        return dtype_from_pylibcudf_lists_column(col)
    elif tid == pylibcudf.TypeId.STRUCT:
        return dtype_from_pylibcudf_structs_column(col)
    elif tid == pylibcudf.TypeId.DECIMAL64:
        return cudf.Decimal64Dtype(
            precision=cudf.Decimal64Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    elif tid == pylibcudf.TypeId.DECIMAL32:
        return cudf.Decimal32Dtype(
            precision=cudf.Decimal32Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    elif tid == pylibcudf.TypeId.DECIMAL128:
        return cudf.Decimal128Dtype(
            precision=cudf.Decimal128Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    else:
        return PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
            <underlying_type_t_type_id>(tid)
        ]
