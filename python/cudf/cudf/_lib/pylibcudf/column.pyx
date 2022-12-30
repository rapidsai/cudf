# Copyright (c) 2022, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool as cbool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.types cimport size_type

from .column_view cimport ColumnView


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
