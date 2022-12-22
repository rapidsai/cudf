# Copyright (c) 2022, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, data_type, size_type, type_id

from .types cimport py_type_to_c_type
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr


cdef class ColumnView:
    """Wrapper around column_view."""
    cdef unique_ptr[column_view] * c_obj

    # TODO: For now assuming data and mask are Buffers, but eventually need to
    # define a new gpumemoryview type to handle this. For that object it should
    # be possible to access all attributes via fast cdef functions (no Python
    # overhead for querying size etc).
    def __cinit__(self, py_type_id, object data_buf, object mask_buf):
        cdef type_id c_type_id = py_type_to_c_type(py_type_id)
        cdef data_type dtype = data_type(c_type_id)
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

    cdef column_view get(self) nogil:
        """Get the underlying column_view object.

        Note that this returns a copy, but by design column_view is designed to
        be lightweight and easy to copy so this is acceptable.
        """
        return dereference(self.c_obj.get())


cdef unique_ptr[column] copy_column(ColumnView col):
    """Deep copies a column

    Parameters
    ----------
    col : ColumnView
        The column to be copied.

    Returns
    -------
    column
        A deep copy of the input column.
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(make_unique[column](col.get()))

    return move(c_result)
