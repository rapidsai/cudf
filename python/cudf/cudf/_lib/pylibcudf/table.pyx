# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table

from . cimport libcudf_classes
from .column cimport Column

# ctypedef _vec_cols vector[unique_ptr[column]]

cdef class Table:
    """A set of columns of the same size."""
    def __init__(self, object columns):
        self.column = columns

        self._underlying = None

    cpdef libcudf_classes.TableView get_underlying(self):
        if self._underlying is None:
            self._underlying = libcudf_classes.TableView(
                [col.get_underlying() for col in self.columns]
            )
        return self._underlying

    # TODO: I'm currently inconsistent between Table and Column how much of the
    # corresponding libcudf_class is being used. I need to make some design
    # decisions about that eventually.
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
