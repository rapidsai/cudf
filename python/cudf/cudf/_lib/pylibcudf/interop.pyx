# Copyright (c) 2023-2024, NVIDIA CORPORATION.

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


cpdef Table from_arrow(pa.Table pyarrow_table):
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

    cdef shared_ptr[pa.CTable] ctable = (
        pa.pyarrow_unwrap_table(pyarrow_table)
    )
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_from_arrow(ctable.get()[0]))

    return Table.from_libcudf(move(c_result))


cpdef pa.Table to_arrow(Table tbl, list metadata):
    """Convert to a PyArrow Table.

    Parameters
    ----------
    metadata : list
        The metadata to attach to the columns of the table.
    """
    cdef shared_ptr[pa.CTable] c_result
    cdef vector[column_metadata] c_metadata
    cdef ColumnMetadata meta
    for meta in metadata:
        c_metadata.push_back(meta.to_libcudf())

    with nogil:
        c_result = move(cpp_to_arrow(tbl.view(), c_metadata))

    return pa.pyarrow_wrap_table(c_result)
