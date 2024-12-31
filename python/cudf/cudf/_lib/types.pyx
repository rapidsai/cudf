# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np

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
    else:
        return cudf.ListDtype(dtype_from_column_view(child))

cdef dtype_from_column_view(column_view cv):
    cdef libcudf_types.type_id tid = cv.type().id()
    if tid == libcudf_types.type_id.LIST:
        return dtype_from_lists_column_view(cv)
    elif tid == libcudf_types.type_id.STRUCT:
        fields = {
            str(i): dtype_from_column_view(cv.child(i))
            for i in range(cv.num_children())
        }
        return cudf.StructDtype(fields)
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
