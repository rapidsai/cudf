# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column

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
            c_result = move(make_unique[column](cv.get()))

        ret.c_obj.swap(c_result)
        return ret
