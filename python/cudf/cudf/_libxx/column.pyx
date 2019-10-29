import numpy as np

from libc.stdint cimport uintptr_t
from libcpp.pair cimport pair
from libcpp cimport bool

from rmm import device_array_from_ptr
from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer
from rmm._lib.device_buffer import DeviceBuffer

from cudf._libxx.lib cimport *
from cudf._libxx.buffer import Buffer

np_to_cudf_types = {np.dtype('int32'): INT32,
                    np.dtype('int64'): INT64,
                    np.dtype('float32'): FLOAT32,
                    np.dtype('float64'): FLOAT64}

cudf_to_np_types = {INT32: np.dtype('int32'),
                    INT64: np.dtype('int64'),
                    FLOAT32: np.dtype('float32'),
                    FLOAT64: np.dtype('float64')}



cdef class _Column:
    def __cinit__(self):
        pass

    @staticmethod
    cdef from_ptr(unique_ptr[column] ptr):
        cdef _Column col = _Column.__new__(_Column)
        col.c_obj = move(ptr)
        return col

    cdef size_type size(self):
        return self.c_obj.get()[0].size()

    cdef data_type type(self):
        return self.c_obj.get()[0].type()

    cpdef bool has_nulls(self):
        return self.c_obj.get()[0].has_nulls()

    @property
    def dtype(self):
        return cudf_to_np_types[self.type().id()]

    def release_into_column(self):
        data = DeviceBuffer.from_ptr(
            ptr=int(<uintptr_t>(self.c_obj.get()[0].view().data[void]())),
            size=self.dtype.itemsize * self.size())

        if self.has_nulls():
            mask = DeviceBuffer.from_ptr(
                ptr=int(<uintptr_t>(self.c_obj.get()[0].view().null_mask())),
                size=self.dtype.itemsize * self.size())
        else:
            mask = None
        return Column(data=data,
                      size=self.size(),
                      dtype=self.dtype,
                      mask=mask)


cdef class Column:
    def __init__(self, data, size, dtype, mask=None):
        self.data = data
        self.size = size
        self.dtype = dtype
        self.mask = mask

    cdef mutable_column_view mutable_view(self):
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

    cdef column_view view(self):
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

