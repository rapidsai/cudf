# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool as cbool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.types cimport size_type

from .column_view cimport ColumnView


cdef class ColumnContents:
    # TODO: Currently treating this like a C POD struct, should consider giving
    # it more real structure.
    pass


cdef class Column:
    """Wrapper around column."""

    # TODO: Would be nice to have this cpdefed, but static cpdef methods are
    # not yet supported. Best option for now may be to have a separate cdef
    # function that is called by the def function.
    @staticmethod
    def from_ColumnView(ColumnView cv):
        """Deep copies a column view's data.

        Parameters
        ----------
        cv : ColumnView
            The column to be copied.

        Returns
        -------
        Column
            A deep copy of the input column.
        """
        ret = Column()

        cdef unique_ptr[column] c_result
        with nogil:
            c_result = move(make_unique[column](dereference(cv.get())))

        ret.c_obj.swap(c_result)
        return ret

    cdef column * get(self) nogil:
        """Get the underlying column object."""
        return self.c_obj.get()

    cpdef size_type size(self):
        return self.get().size()

    cpdef size_type null_count(self):
        return self.get().null_count()

    cpdef cbool has_nulls(self):
        return self.get().has_nulls()

    cpdef ColumnView view(self):
        return ColumnView.from_column_view(self.get().view())

    cpdef ColumnContents release(self):
        """Release the data in this column.

        After this method is called, any usage of this object will lead to seg
        faults.
        """
        # TODO: Consider implementing a safety flag to prevent performing any
        # operations on a released Column. Using a c bool flag it should be
        # basically free (although repetitive) to do.
        cdef column_contents contents = move(self.get().release())
        cdef ColumnContents ret = ColumnContents()
        ret.data = DeviceBuffer.c_from_unique_ptr(move(contents.data))
        ret.null_mask = DeviceBuffer.c_from_unique_ptr(
            move(
                contents.null_mask
            )
        )
        # TODO: Implement for children (currently they are discarded).
        # cdef
        return ret
