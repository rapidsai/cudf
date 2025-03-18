# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table

from .column cimport Column

__all__ = ["Table"]

cdef class Table:
    """A list of columns of the same size.

    Parameters
    ----------
    columns : list
        The columns in this table.
    """
    def __init__(self, list columns):
        if not all(isinstance(c, Column) for c in columns):
            raise ValueError("All columns must be pylibcudf Column objects")
        self._columns = columns

    __hash__ = None

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
        cdef vector[unique_ptr[column]] c_columns = dereference(libcudf_tbl).release()

        cdef vector[unique_ptr[column]].size_type i
        return Table([
            Column.from_libcudf(move(c_columns[i]))
            for i in range(c_columns.size())
        ])

    @staticmethod
    cdef Table from_table_view(const table_view& tv, Table owner):
        """Create a Table from a libcudf table_view into a Table owner.

        This method accepts shared ownership of the underlying data from the owner.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef int i
        return Table([
            Column.from_column_view(tv.column(i), owner.columns()[i])
            for i in range(tv.num_columns())
        ])

    # Ideally this function would simply be handled via a fused type in
    # from_table_view, but this does not work due to
    # https://github.com/cython/cython/issues/6740
    @staticmethod
    cdef Table from_table_view_of_arbitrary(const table_view& tv, object owner):
        """Create a Table from a libcudf table_view into an arbitrary owner.

        This method accepts shared ownership of the underlying data from the owner.
        Since the owner may be any arbitrary object, every buffer view that is part of
        the table (each column and all of their children) share ownership of the same
        buffer since they do not have the information available to choose to only own
        subsets of it.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # For efficiency, prohibit calling this overload with a Table owner.
        assert not isinstance(owner, Table)
        cdef int i
        return Table([
            Column.from_column_view_of_arbitrary(tv.column(i), owner)
            for i in range(tv.num_columns())
        ])

    cpdef int num_columns(self):
        """The number of columns in this table."""
        return len(self._columns)

    cpdef int num_rows(self):
        """The number of rows in this table."""
        if self.num_columns() == 0:
            return 0
        return self._columns[0].size()

    cpdef list columns(self):
        """The columns in this table."""
        return self._columns
