import numpy as np

from libc.stdint cimport uintptr_t

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


cdef class Column:
    def __cinit__(self, data, size, dtype, mask=None):
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
            data = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            data = NULL
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
            data = <bitmask_type*><uintptr_t>(self.mask.ptr)
        else:
            data = NULL
        return column_view(
            dtype,
            self.size,
            data,
            mask)
        

cdef class _Column:
    def __cinit__(self):
        pass

    @property
    def data(self):
        """
        Return the underlying data as a `Buffer` whose lifetime
        is tied to the column itself.
        """
        return Buffer(
            ptr=self.c_obj[0].view().data(),
            size=self.c_obj[0].size(),
            owner=self)

    @property
    def mask(self):
        """
        Return the underlying mask as a `Buffer` whose lifetime
        is tied to the column itself.
        """
        return Buffer(
            ptr=self.c_obj[0].view().null_mask(),
            size=self.c_obj[0].size(),
            owner=self)
