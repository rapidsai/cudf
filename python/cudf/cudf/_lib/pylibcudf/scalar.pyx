# Copyright (c) 2023, NVIDIA CORPORATION.

cimport pyarrow.lib
from cython cimport no_gc_clear
from cython.operator cimport dereference
from libcpp.utility cimport move

import pyarrow.lib

from rmm._lib.memory_resource cimport get_current_device_resource

from cudf._lib.cpp cimport aggregation
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.reduce cimport cpp_reduce
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type

from .column cimport Column
from .interop cimport from_arrow
from .table cimport Table
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

    def __init__(self, pyarrow.lib.Scalar value=None):
        self.mr = get_current_device_resource()

        if value is None:
            # TODO: This early return is not something we really want to
            # support, but it here for now to ease the transition of
            # DeviceScalar.
            return

        # Convert the value to a cudf object via pyarrow
        arr = pyarrow.lib.array([value.as_py()], type=value.type)
        cdef pyarrow.lib.Table pa_tbl = pyarrow.lib.Table.from_arrays(
            [arr], names=["scalar"]
        )
        cdef Table tbl = from_arrow(pa_tbl)

        # TODO: The code below is a pretty hacky way to get a scalar from a
        # single row of a column, but for now want to see if we can write a
        # generic solution like this that works. If it does, we can consider
        # implementing a better approach natively in libcudf.
        cdef Column col = tbl.columns()[0]
        cdef column_view cv = col.view()

        cdef unique_ptr[scalar] c_result
        cdef unique_ptr[aggregation.reduce_aggregation] c_agg = (
            aggregation.make_min_aggregation[aggregation.reduce_aggregation]()
        )
        cdef data_type c_type = col.type().c_obj

        with nogil:
            c_result = move(cpp_reduce(
                cv,
                dereference(c_agg),
                c_type
            ))

        cdef Scalar s = Scalar.from_libcudf(move(c_result))
        self._data_type = DataType.from_libcudf(s.get().type())
        self.c_obj.swap(s.c_obj)

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
