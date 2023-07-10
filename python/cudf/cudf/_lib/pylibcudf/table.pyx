# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table

from .column cimport Column


cdef class Table:
    """A set of columns of the same size."""
    def __init__(self, object columns):
        self.columns = columns

    cdef table_view view(self) nogil:
        # TODO: Make c_columns a class attribute that is updated along with
        # self.columns whenever new columns are added or columns are removed.
        cdef vector[column_view] c_columns

        with gil:
            for col in self.columns:
                c_columns.push_back((<Column> col).view())

        return table_view(c_columns)

    @staticmethod
    cdef Table from_libcudf(unique_ptr[table] libcudf_tbl):
        cdef vector[unique_ptr[column]] c_columns = move(
            dereference(libcudf_tbl).release()
        )

        cdef vector[unique_ptr[column]].size_type i
        return Table([
            Column.from_libcudf(move(c_columns[i]))
            for i in range(c_columns.size())
        ])
