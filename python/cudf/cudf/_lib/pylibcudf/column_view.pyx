# Copyright (c) 2022, NVIDIA CORPORATION.

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, data_type, size_type, type_id


# Helpers
# These should be extracted into separate modules.
cdef void * int_to_void_ptr(ptr):
    # Cython will not cast a Python integer directly to a pointer, so the
    # intermediate cast to a uintptr_t is necessary
    return <void*><uintptr_t>(ptr)

cdef bitmask_type * int_to_bitmask_ptr(ptr):
    # Cython will not cast a Python integer directly to a pointer, so the
    # intermediate cast to a uintptr_t is necessary
    return <bitmask_type*><uintptr_t>(ptr)

cdef class ColumnView:
    """Wrapper around column_view."""
    cdef unique_ptr[column_view] * c_obj

    def __cinit__(self, object data_buf, object mask_buf):
        # TODO: For now assuming data is a Buffer, but eventually need to
        # define a new gpumemoryview type to handle this. For that object it
        # should be possible to access all attributes via fast cdef functions
        # (no Python overhead for querying size etc).
        cdef data_type dtype = data_type(type_id.INT32)
        cdef size_type size = data_buf.size
        cdef const void * data = int_to_void_ptr(data_buf.ptr)
        cdef const bitmask_type * null_mask
        if mask_buf is not None:
            null_mask = int_to_bitmask_ptr(mask_buf.ptr)
        # TODO: At the moment libcudf does not expose APIs for counting the
        # nulls in a bitmask directly (those APIs are in detail/null_mask). If
        # we want to allow more flexibility in the Cython layer we'll need to
        # expose those eventually. This dovetails with our desire to expose
        # other functionality too like bitmask_and.
        cdef size_type null_count = 0
        # TODO: offset and children not yet supported
        cdef size_type offset = 0
        cdef const vector[column_view] children

        self.c_obj.reset(
            new column_view(
                dtype, size, data, null_mask, null_count, offset, children
            )
        )
