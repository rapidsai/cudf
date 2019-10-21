import numpy as np

from libc.stdint cimport uintptr_t

from rmm import device_array_from_ptr
from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer
from rmm._lib.device_buffer import DeviceBuffer

from cudf._libxx.lib cimport *


np_to_cudf_types = {np.dtype('int32'): INT32,
                    np.dtype('int64'): INT64,
                    np.dtype('float32'): FLOAT32,
                    np.dtype('float64'): FLOAT64}

cudf_to_np_types = {INT32: np.dtype('int32'),
                    INT64: np.dtype('int64'),
                    FLOAT32: np.dtype('float32'),
                    FLOAT64: np.dtype('float64')}

cdef class Column:
    def __cinit__(self):
        self.c_obj = new column()

    @classmethod
    def from_array(cls, array):
        cdef Column col = Column.__new__(Column)
        cdef type_id dtype = np_to_cudf_types[array.dtype]
        buf = DeviceBuffer(array)
        col.c_obj = new column(
            data_type(dtype),
            len(array),
            buf.c_obj)
        return col

    def _view(self):
        cdef ColumnView cview = ColumnView.__new__(ColumnView, owner=self)
        cview.c_obj[0] = self.c_obj.view()
        return cview

    def _mutable_view(self):
        cdef MutableColumnView mcview = MutableColumnView.__new__(
            MutableColumnView, owner=self)
        mcview.c_obj[0] = self.c_obj.mutable_view()
        return mcview

    def view(self, readonly=False):
        if readonly:
            return self._view()
        else:
            return self._mutable_view()

    def size(self):
        return self.c_obj[0].size()

    def __dealloc__(self):
        del self.c_obj


cdef class MutableColumnView:
    def __cinit__(self, owner):
        self.owner = owner
        self.c_obj = new mutable_column_view()

    def __dealloc__(self):
        del self.c_obj

    def dtype(self):
        return cudf_to_np_types[self.c_obj[0].type().id()]

    def size(self):
        return self.c_obj[0].size()

    def nullable(self):
        return self.c_obj[0].nullable()

    def has_nulls(self):
        return self.c_obj[0].has_nulls()

    def offset(self):
        return self.c_obj[0].offset()

    def gpu_array_view(self):
        cdef uintptr_t ptr = <uintptr_t>(self.c_obj[0].data[void]())
        return device_array_from_ptr(
            ptr,
            self.size(),
            self.dtype())


cdef class ColumnView:
    def __cinit__(self, owner):
        self.owner = owner
        self.c_obj = new column_view()

    def __dealloc__(self):
        del self.c_obj

    def dtype(self):
        return cudf_to_np_types[self.c_obj[0].type().id()]

    def size(self):
        return self.c_obj[0].size()

    def nullable(self):
        return self.c_obj[0].nullable()

    def has_nulls(self):
        return self.c_obj[0].has_nulls()

    def offset(self):
        return self.c_obj[0].offset()

    def gpu_array_view(self):

        cdef uintptr_t ptr = <uintptr_t>(self.c_obj[0].data[void]())
        return device_array_from_ptr(
            ptr,
            self.size(),
            self.dtype())
