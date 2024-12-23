# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from libcpp.memory cimport make_shared, shared_ptr

cimport pylibcudf.libcudf.types as libcudf_types
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view

import pylibcudf as plc

import cudf


SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES = {
    np.dtype("int8"): plc.types.TypeId.INT8,
    np.dtype("int16"): plc.types.TypeId.INT16,
    np.dtype("int32"): plc.types.TypeId.INT32,
    np.dtype("int64"): plc.types.TypeId.INT64,
    np.dtype("uint8"): plc.types.TypeId.UINT8,
    np.dtype("uint16"): plc.types.TypeId.UINT16,
    np.dtype("uint32"): plc.types.TypeId.UINT32,
    np.dtype("uint64"): plc.types.TypeId.UINT64,
    np.dtype("float32"): plc.types.TypeId.FLOAT32,
    np.dtype("float64"): plc.types.TypeId.FLOAT64,
    np.dtype("datetime64[s]"): plc.types.TypeId.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): plc.types.TypeId.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): plc.types.TypeId.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): plc.types.TypeId.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): plc.types.TypeId.STRING,
    np.dtype("bool"): plc.types.TypeId.BOOL8,
    np.dtype("timedelta64[s]"): plc.types.TypeId.DURATION_SECONDS,
    np.dtype("timedelta64[ms]"): plc.types.TypeId.DURATION_MILLISECONDS,
    np.dtype("timedelta64[us]"): plc.types.TypeId.DURATION_MICROSECONDS,
    np.dtype("timedelta64[ns]"): plc.types.TypeId.DURATION_NANOSECONDS,
}
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES = {
    plc_type: np_type
    for np_type, plc_type in SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES.items()
}
# There's no equivalent to EMPTY in cudf.  We translate EMPTY
# columns from libcudf to ``int8`` columns of all nulls in Python.
# ``int8`` is chosen because it uses the least amount of memory.
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.EMPTY] = np.dtype("int8")
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.STRUCT] = np.dtype("object")
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.LIST] = np.dtype("object")


size_type_dtype = PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.SIZE_TYPE_ID]


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
        return PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
            <underlying_type_t_type_id>(tid)
        ]


cpdef dtype_to_pylibcudf_type(dtype):
    if isinstance(dtype, cudf.ListDtype):
        return plc.DataType(plc.TypeId.LIST)
    elif isinstance(dtype, cudf.StructDtype):
        return plc.DataType(plc.TypeId.STRUCT)
    elif isinstance(dtype, cudf.Decimal128Dtype):
        tid = plc.TypeId.DECIMAL128
        return plc.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal64Dtype):
        tid = plc.TypeId.DECIMAL64
        return plc.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal32Dtype):
        tid = plc.TypeId.DECIMAL32
        return plc.DataType(tid, -dtype.scale)
    # libcudf types don't support timezones so convert to the base type
    elif isinstance(dtype, pd.DatetimeTZDtype):
        dtype = np.dtype(f"<M8[{dtype.unit}]")
    else:
        dtype = np.dtype(dtype)
    return plc.DataType(SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES[dtype])


def dtype_from_pylibcudf_lists_column(col):
    child = col.list_view().child()
    tid = child.type().id()

    if tid == plc.TypeId.LIST:
        return cudf.ListDtype(dtype_from_pylibcudf_lists_column(child))
    elif tid == plc.TypeId.EMPTY:
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

    if tid == plc.TypeId.LIST:
        return dtype_from_pylibcudf_lists_column(col)
    elif tid == plc.TypeId.STRUCT:
        return dtype_from_pylibcudf_structs_column(col)
    elif tid == plc.TypeId.DECIMAL64:
        return cudf.Decimal64Dtype(
            precision=cudf.Decimal64Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    elif tid == plc.TypeId.DECIMAL32:
        return cudf.Decimal32Dtype(
            precision=cudf.Decimal32Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    elif tid == plc.TypeId.DECIMAL128:
        return cudf.Decimal128Dtype(
            precision=cudf.Decimal128Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    else:
        return PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[tid]
