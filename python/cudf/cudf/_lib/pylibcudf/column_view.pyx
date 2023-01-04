# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from libcpp.vector cimport vector

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, size_type

from .types cimport DataType
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr


cdef class ColumnView:
    """Wrapper around column_view."""
    # TODO: For now assuming data and mask are Buffers, but eventually need to
    # define a new gpumemoryview type to handle this. For that object it should
    # be possible to access all attributes via fast cdef functions (no Python
    # overhead for querying size etc).
    # TODO: Not currently supporting SpillableBuffers.
    # TODO: Need a way to map the data buffer size to the number of
    # elements. For fixed width types a mapping could be made based on the
    # number of bytes they occupy, but not for nested types. Not sure how
    # best to expose that in the API yet, but matching C++ for now and
    # requesting the size from the user. The gpumemoryview may also help.
    # TODO: Should be using `not None` where possible.
    # TODO: I've temporarily defined __init__ instead of __cinit__ so that
    # factory functions can call __new__ without arguments. I'll need to think
    # more fully about what construction patterns we actually want to support.
    def __init__(
        self, DataType dtype, size_type size, object data_buf, object mask_buf
    ):
        # TODO: Investigate cases where the data_buf is None. I'm not sure that
        # this is a real use case that we should support.
        cdef const void * data = NULL
        if data_buf is not None:
            data = int_to_void_ptr(data_buf.ptr)
        cdef const bitmask_type * null_mask = NULL
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
                dtype.c_obj, size, data, null_mask, null_count, offset,
                children
            )
        )

    cdef column_view * get(self) nogil:
        """Get the underlying column_view object."""
        return self.c_obj.get()

    # TODO: Unclear if this needs to be exposed in the Python API or if Cython
    # is sufficient.
    @staticmethod
    cdef from_column_view(column_view cv):
        cdef ColumnView ret = ColumnView.__new__(ColumnView)
        ret.c_obj.reset(new column_view(cv))
        return ret

    cpdef size_type size(self):
        return self.get().size()

    cpdef size_type null_count(self):
        return self.get().null_count()
