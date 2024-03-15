# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pyarrow cimport lib as pa

from cudf._lib.cpp.interop cimport (
    column_metadata,
    from_arrow as cpp_from_arrow,
    to_arrow as cpp_to_arrow,
)
from cudf._lib.cpp.table.table cimport table

from .table cimport Table


cdef class ColumnMetadata:
    """Metadata associated with a column.

    This is the Cython representation of :cpp:class:`cudf::column_metadata`.

    Parameters
    ----------
    id : TypeId
        The type's identifier
    scale : int
        The scale associated with the data. Only used for decimal data types.
    """
    def __init__(self, name):
        self.name = name
        self.children_meta = []

    cdef column_metadata to_libcudf(self):
        """Convert to C++ column_metadata.

        Since this class is mutable and cheap, it is easier to create the C++
        object on the fly rather than have it directly backing the storage for
        the Cython class.
        """
        cdef column_metadata c_metadata
        cdef ColumnMetadata child_meta
        c_metadata.name = self.name.encode()
        for child_meta in self.children_meta:
            c_metadata.children_meta.push_back(child_meta.to_libcudf())
        return c_metadata


# These functions are pure Python functions in anticipation of when we no
# longer use pyarrow's Cython and instead just leverage the capsule interface.
def from_arrow(pyarrow_object):
    """Create a Table from a PyArrow Table.

    Parameters
    ----------
    pyarrow_table : pyarrow.Table
        The PyArrow Table to convert to a Table.

    Returns
    -------
    Table
        The converted Table.
    """
    cdef shared_ptr[pa.CTable] arrow_table
    cdef unique_ptr[table] c_table_result

    if isinstance(pyarrow_object, pa.Table):
        arrow_table = pa.pyarrow_unwrap_table(pyarrow_object)

        with nogil:
            c_table_result = move(cpp_from_arrow(dereference(arrow_table)))

        return Table.from_libcudf(move(c_table_result))

    raise TypeError("from_arrow only accepts pyarrow.Table objects")


def to_arrow(cudf_object, list metadata=None):
    """Convert to a PyArrow Table.

    Parameters
    ----------
    metadata : list
        The metadata to attach to the columns of the table.
    """
    cdef shared_ptr[pa.CTable] c_table_result
    cdef vector[column_metadata] c_metadata
    cdef ColumnMetadata meta

    if isinstance(cudf_object, Table):
        for meta in metadata:
            c_metadata.push_back(meta.to_libcudf())

        with nogil:
            c_table_result = move(
                cpp_to_arrow((<Table> cudf_object).view(), c_metadata)
            )

        return pa.pyarrow_wrap_table(c_table_result)

    raise TypeError("to_arrow only accepts Table objects")
