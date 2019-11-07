import numpy as np

import rmm

from libc.stdint cimport uintptr_t
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._libxx.lib cimport *
from cudf.core.buffer import Buffer
from libc.stdlib cimport malloc, free

from cudf.utils.dtypes import is_categorical_dtype

np_to_cudf_types = {np.dtype('int8'): INT8,
                    np.dtype('int16'): INT16,
                    np.dtype('int32'): INT32,
                    np.dtype('int64'): INT64,
                    np.dtype('float32'): FLOAT32,
                    np.dtype('float64'): FLOAT64,
                    np.dtype("datetime64[D]"): TIMESTAMP_DAYS,
                    np.dtype("datetime64[s]"): TIMESTAMP_SECONDS,
                    np.dtype("datetime64[ms]"): TIMESTAMP_MILLISECONDS,
                    np.dtype("datetime64[us]"): TIMESTAMP_MICROSECONDS,
                    np.dtype("datetime64[ns]"): TIMESTAMP_NANOSECONDS,
                    np.dtype("object"): STRING,
}

cudf_to_np_types = {INT8: np.dtype('int8'),
                    INT16: np.dtype('int16'),
                    INT32: np.dtype('int32'),
                    INT64: np.dtype('int64'),
                    FLOAT32: np.dtype('float32'),
                    FLOAT64: np.dtype('float64'),
                    TIMESTAMP_DAYS: np.dtype("datetime64[D]"),
                    TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
                    TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
                    TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
                    TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
                    STRING: np.dtype("object")
}

cdef class Column:
    def __init__(self, data, size, dtype, mask=None):
        self.data = data
        self.size = size
        self.dtype = dtype
        self.mask = mask

    @property
    def null_count(self):
        return self.null_count()

    cdef size_type null_count(self):
        return self.view().null_count()

    cdef mutable_column_view mutable_view(self) except *:
        cdef type_id tid = np_to_cudf_types[np.dtype(self.dtype)]
        cdef data_type dtype = data_type(tid)
        cdef void* data = <void*><uintptr_t>(self.data.ptr)
        cdef bitmask_type* mask
        if self.mask is not None:
            mask = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            mask = NULL
        return mutable_column_view(
            dtype,
            self.size,
            data,
            mask)

    cdef column_view view(self) except *:
        cdef type_id tid = np_to_cudf_types[np.dtype(self.dtype)]
        cdef data_type dtype = data_type(tid)
        cdef void* data = <void*><uintptr_t>(self.data.ptr)
        cdef bitmask_type* mask
        if self.mask is not None:
            mask = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            mask = NULL
        return column_view(
            dtype,
            self.size,
            data,
            mask)

    @staticmethod
    cdef Column from_ptr(unique_ptr[column] c_col):
        from cudf.core.column import build_column

        size = c_col.get()[0].size()
        dtype = cudf_to_np_types[c_col.get()[0].type().id()]
        has_nulls = c_col.get()[0].has_nulls()
        cdef column_contents contents = c_col.get()[0].release()
        data = DeviceBuffer.from_ptr(contents.data.release())
        if has_nulls:
            mask = DeviceBuffer.from_ptr(contents.null_mask.release())
        else:
            mask = None
        return build_column(data, dtype=dtype, mask=mask)
