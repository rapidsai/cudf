# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.contiguous_split cimport (
    pack as cpp_pack,
    packed_columns,
    unpack as cpp_unpack,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view

from .table cimport Table


cdef class PackedColumns:
    """Column data in a serialized format.

    Contains data from an array of columns in two contiguous buffers:
    one on host, which contains table metadata and one on device which
    contains the table data.

    For details, see :cpp:class:`cudf::packed_columns`.
    """
    def __init__(self):
        raise ValueError(
            "PackedColumns should not be constructed directly. "
            "Use one of the factories."
        )

    @staticmethod
    cdef PackedColumns from_libcudf(unique_ptr[packed_columns] data):
        """Create a Python PackedColumns from a libcudf packed_columns."""
        cdef PackedColumns out = PackedColumns.__new__(PackedColumns)
        out.c_obj = move(data)
        return out

cpdef PackedColumns pack(Table input):
    """Deep-copy a table into a serialized contiguous memory format.

    For details, see :cpp:func:`cudf::pack`.

    Parameters
    ----------
    input : Table
        Table to pack.

    Returns
    -------
    PackedColumns
        The packed columns.
    """
    return PackedColumns.from_libcudf(
        make_unique[packed_columns](cpp_pack(input.view()))
    )


cpdef Table unpack(PackedColumns input):
    """Deserialize the result of `pack`.

    Copies the result of a serialized table into a table.
    Contrary to the libcudf C++ function, the returned table is a copy
    of the serialized data.

    For details, see :cpp:func:`cudf::unpack`.

    Parameters
    ----------
    input : PackedColumns
        The packed columns to unpack.

    Returns
    -------
    Table
        Copy of the packed columns.
    """
    cdef table_view v = cpp_unpack(dereference(input.c_obj))
    cdef unique_ptr[table] t = make_unique[table](v)
    return Table.from_libcudf(move(t))
