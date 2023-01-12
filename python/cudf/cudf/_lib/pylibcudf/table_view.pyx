# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view

# from .gpumemoryview cimport gpumemoryview
# from .types cimport DataType
# from .utils cimport int_to_bitmask_ptr, int_to_void_ptr
from .column_view cimport ColumnView

# from cudf._lib.cpp.types cimport bitmask_type, size_type


cdef class TableView:
    """Wrapper around table_view."""
    # TODO: Could accept an arbitrary iterable of columns, but that will be
    # less performant and harder to type-check.
    def __init__(self, list columns):
        self.columns = [c for c in columns]
        cdef vector[column_view] c_columns
        cdef ColumnView col
        for col in self.columns:
            c_columns.push_back(dereference(col.get()))

        self.c_obj.reset(new table_view(c_columns))

    cdef table_view * get(self) nogil:
        """Get the underlying table_view object."""
        return self.c_obj.get()

    cpdef size_type num_columns(self):
        return self.get().num_columns()

    cpdef size_type num_rows(self):
        return self.get().num_rows()
