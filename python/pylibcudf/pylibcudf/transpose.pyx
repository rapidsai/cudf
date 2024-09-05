# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from pylibcudf.libcudf cimport transpose as cpp_transpose
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table_view cimport table_view

from .column cimport Column
from .table cimport Table


cpdef tuple transpose(Table input_table):
    """Transpose a Table.

    For details, see :cpp:func:`transpose`.

    Parameters
    ----------
    input_table : Table
        Table to transpose

    Returns
    -------
    tuple[Column, Table]
        Two-tuple transposed column and table.
    """
    cdef pair[unique_ptr[column], table_view] c_result

    with nogil:
        c_result = move(cpp_transpose.transpose(input_table.view()))

    return (
        Column.from_libcudf(move(c_result.first)),
        Table.from_table_view(c_result.second, input_table)
    )
