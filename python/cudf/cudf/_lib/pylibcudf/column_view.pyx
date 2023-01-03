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
    def __cinit__(
        self, DataType dtype, size_type size, object data_buf, object mask_buf
    ):
        # TODO: Can the data_buf be None? We currently allow for that in cudf
        # when a Column's base_data is None, but I don't know why. I think that
        # should be filtered out upstream of here.
        # don't know if that's actually support
        cdef const void * data = int_to_void_ptr(data_buf.ptr)
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
