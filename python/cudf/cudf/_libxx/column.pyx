import numpy as np

from libc.stdint cimport uintptr_t
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._libxx.lib cimport *
from cudf.core.buffer import Buffer

np_to_cudf_types = {np.dtype('int32'): INT32,
                    np.dtype('int64'): INT64,
                    np.dtype('float32'): FLOAT32,
                    np.dtype('float64'): FLOAT64}

cudf_to_np_types = {INT32: np.dtype('int32'),
                    INT64: np.dtype('int64'),
                    FLOAT32: np.dtype('float32'),
                    FLOAT64: np.dtype('float64')}


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
        return mutable_column_views(
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
        size = c_col.get()[0].size()
        dtype = cudf_to_np_types[c_col.get()[0].type().id()]
        has_nulls = c_col.get()[0].has_nulls()
        cdef column_contents contents = c_col.get()[0].release()
        data = DeviceBuffer.from_ptr(contents.data.release())
        if has_nulls:
            mask = DeviceBuffer.from_ptr(contents.null_mask.release())
        else:
            mask = None
        return Column(data, size=size, dtype=dtype, mask=mask)
