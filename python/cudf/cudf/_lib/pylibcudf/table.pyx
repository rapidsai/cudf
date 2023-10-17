# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pyarrow cimport lib as pa

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.interop cimport (
    column_metadata,
    from_arrow as cpp_from_arrow,
    to_arrow as cpp_to_arrow,
)
from cudf._lib.cpp.table.table cimport table

from .column cimport Column
from .interop cimport ColumnMetadata


cdef class Table:
    """A list of columns of the same size.

    Parameters
    ----------
    columns : list
        The columns in this table.
    """
    def __init__(self, list columns):
        self._columns = columns

    cdef table_view view(self) nogil:
        """Generate a libcudf table_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # TODO: Make c_columns a class attribute that is updated along with
        # self._columns whenever new columns are added or columns are removed.
        cdef vector[column_view] c_columns

        with gil:
            for col in self._columns:
                c_columns.push_back((<Column> col).view())

        return table_view(c_columns)

    @staticmethod
    cdef Table from_libcudf(unique_ptr[table] libcudf_tbl):
        """Create a Table from a libcudf table.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef vector[unique_ptr[column]] c_columns = move(
            dereference(libcudf_tbl).release()
        )

        cdef vector[unique_ptr[column]].size_type i
        return Table([
            Column.from_libcudf(move(c_columns[i]))
            for i in range(c_columns.size())
        ])

    cpdef list columns(self):
        return self._columns

    @staticmethod
    def from_arrow(pa.Table pyarrow_table):
        cdef shared_ptr[pa.CTable] ctable = (
            pa.pyarrow_unwrap_table(pyarrow_table)
        )
        cdef unique_ptr[table] c_result

        with nogil:
            c_result = move(cpp_from_arrow(ctable.get()[0]))

        return Table.from_libcudf(move(c_result))

    cpdef pa.Table to_arrow(self, list metadata):
        cdef shared_ptr[pa.CTable] c_result
        cdef vector[column_metadata] c_metadata
        cdef ColumnMetadata meta
        for meta in metadata:
            c_metadata.push_back(meta.to_libcudf())

        with nogil:
            c_result = move(cpp_to_arrow(self.view(), c_metadata))

        return pa.pyarrow_wrap_table(c_result)
