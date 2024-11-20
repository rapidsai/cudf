# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from pylibcudf.libcudf cimport transpose as cpp_transpose
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table_view cimport table_view

from .column cimport Column
from .table cimport Table

__all__ = ["transpose"]

cpdef Table transpose(Table input_table):
    """Transpose a Table.

    For details, see :cpp:func:`transpose`.

    Parameters
    ----------
    input_table : Table
        Table to transpose

    Returns
    -------
    Table
        Transposed table.
    """
    cdef pair[unique_ptr[column], table_view] c_result
    cdef Table owner_table

    with nogil:
        c_result = cpp_transpose.transpose(input_table.view())

    owner_table = Table(
        [Column.from_libcudf(move(c_result.first))] * c_result.second.num_columns()
    )

    return Table.from_table_view(c_result.second, owner_table)
