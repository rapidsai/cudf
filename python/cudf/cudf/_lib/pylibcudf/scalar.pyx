# Copyright (c) 2023, NVIDIA CORPORATION.

cimport cython

from rmm._lib.memory_resource cimport get_current_device_resource

from .types cimport DataType


# The DeviceMemoryResource attribute could be released prematurely
# by the gc if the DeviceScalar is in a reference cycle. Removing
# the tp_clear function with the no_gc_clear decoration prevents that.
# See https://github.com/rapidsai/rmm/pull/931 for details.
@cython.no_gc_clear
cdef class Scalar:

    def __cinit__(self, *args, **kwargs):
        self.mr = get_current_device_resource()

    cdef const scalar* get(self) except *:
        return self.c_obj.get()

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef bool is_valid(self):
        """True if the scalar is valid, false if not"""
        return self.get().is_valid()
