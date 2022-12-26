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
    # TODO: For now assuming data and mask are Buffers, but eventually need to
    # define a new gpumemoryview type to handle this. For that object it should
    # be possible to access all attributes via fast cdef functions (no Python
    # overhead for querying size etc).
    # TODO: Need a way to map the data buffer size to the number of
    # elements. For fixed width types a mapping could be made based on the
    # number of bytes they occupy, but not for nested types. Not sure how
    # best to expose that in the API yet, but matching C++ for now and
    # requesting the size from the user. The gpumemoryview may also help.
    def __cinit__(
        self, py_type_id, size_type size, object data_buf, object mask_buf
    ):
        cdef type_id c_type_id = py_type_to_c_type(py_type_id)
        cdef data_type dtype = data_type(c_type_id)
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
                dtype, size, data, null_mask, null_count, offset, children
            )
        )

    cdef column_view get(self) nogil:
        """Get the underlying column_view object.

        Note that this returns a copy, but by design column_view is designed to
        be lightweight and easy to copy so this is acceptable.
        """
        return dereference(self.c_obj.get())


cdef class Column:
    """Wrapper around column."""

    @staticmethod
    def from_column_view(ColumnView cv):
        """Deep copies a column view's data.

        Parameters
        ----------
        col : ColumnView
            The column to be copied.

        Returns
        -------
        column
            A deep copy of the input column.
        """
        ret = Column()

        cdef unique_ptr[column] c_result
        with nogil:
            c_result = move(make_unique[column](cv.get()))

        ret.c_obj.swap(c_result)
        return ret
