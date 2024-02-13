# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp cimport concatenate as cpp_concatenate
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view

from .column cimport Column
from .table cimport Table


cpdef concatenate(list objects):
    """Concatenate columns or tables.

    Parameters
    ----------
    objects : List[Union[Column, Table]]
        The list of Columns or Tables to concatenate.

    Returns
    -------
    Union[Column, Table]
        The concatenated Column or Table.
    """
    cdef vector[column_view] c_columns
    cdef vector[table_view] c_tables

    cdef Column col
    cdef Table tbl

    cdef unique_ptr[column] c_col_result
    cdef unique_ptr[table] c_tbl_result

    cdef int i
    if isinstance(objects[0], Table):
        for i in range(len(objects)):
            tbl = objects[i]
            c_tables.push_back(tbl.view())

        with nogil:
            c_tbl_result = move(cpp_concatenate.concatenate(c_tables))
        return Table.from_libcudf(move(c_tbl_result))
    elif isinstance(objects[0], Column):
        for i in range(len(objects)):
            col = objects[i]
            c_columns.push_back(col.view())

        with nogil:
            c_col_result = move(cpp_concatenate.concatenate(c_columns))
        return Column.from_libcudf(move(c_col_result))
    else:
        raise ValueError("input must be a list of Columns or Tables")
