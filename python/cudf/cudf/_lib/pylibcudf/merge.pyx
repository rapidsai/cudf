# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf cimport merge as cpp_merge
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport null_order, order, size_type

from .table cimport Table


cpdef Table merge (
    list tables_to_merge,
    list key_cols,
    list column_order,
    list null_precedence,
):
    """Merge a set of sorted tables.

    Parameters
    ----------
    tables_to_merge : list
        List of tables to merge.
    key_cols : list
        List of column indexes to merge on.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The merged table.
    """
    cdef vector[size_type] c_key_cols = key_cols
    cdef vector[order] c_column_order = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    cdef vector[table_view] c_tables_to_merge

    for tbl in tables_to_merge:
        c_tables_to_merge.push_back((<Table?> tbl).view())

    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_merge.merge(
                c_tables_to_merge,
                c_key_cols,
                c_column_order,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))
