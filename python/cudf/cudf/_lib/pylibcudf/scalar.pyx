# Copyright (c) 2023, NVIDIA CORPORATION.

cimport pyarrow.lib
from cython cimport no_gc_clear

import pyarrow.lib

from rmm._lib.memory_resource cimport get_current_device_resource

from cudf._lib.cpp.scalar.scalar cimport scalar

from .types cimport DataType


# The DeviceMemoryResource attribute could be released prematurely
# by the gc if the DeviceScalar is in a reference cycle. Removing
# the tp_clear function with the no_gc_clear decoration prevents that.
# See https://github.com/rapidsai/rmm/pull/931 for details.
@no_gc_clear
cdef class Scalar:
    """A scalar value in device memory."""
    # Unlike for columns, libcudf does not support scalar views. All APIs that
    # accept scalar values accept references to the owning object rather than a
    # special view type. As a result, pylibcudf.Scalar has a simpler structure
    # than pylibcudf.Column because it can be a true wrapper around a libcudf
    # column

    def __cinit__(self, *args, **kwargs):
        self.mr = get_current_device_resource()

    def __init__(self, pyarrow.lib.Scalar value=None):
        # TODO: This case is not something we really want to
        # support, but it here for now to ease the transition of
        # DeviceScalar.
        if value is not None:
            raise ValueError("Scalar should be constructed with a factory")

    @staticmethod
    def from_pyarrow_scalar(pyarrow.lib.Scalar value):
        # Need a local import here to avoid a circular dependency because
        # from_arrow_scalar returns a Scalar.
        from .interop import from_arrow_scalar

        return from_arrow_scalar(value)

    cdef const scalar* get(self) except *:
        return self.c_obj.get()

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef bool is_valid(self):
        """True if the scalar is valid, false if not"""
        return self.get().is_valid()

    @staticmethod
    cdef Scalar from_libcudf(unique_ptr[scalar] libcudf_scalar, dtype=None):
        """Construct a Scalar object from a libcudf scalar.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_obj.swap(libcudf_scalar)
        s._data_type = DataType.from_libcudf(s.get().type())
        return s
