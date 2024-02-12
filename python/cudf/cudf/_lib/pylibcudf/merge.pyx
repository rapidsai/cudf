# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp cimport merge as cpp_merge
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport null_order, order, size_type

from .table cimport Table


cpdef Table merge (
    list tables_to_merge,
    list key_cols,
    list column_order,
    list null_precedence,
):
    cdef vector[size_type] c_key_cols = key_cols
    cdef vector[order] c_column_order = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    cdef vector[table_view] c_tables_to_merge

    cdef int i
    cdef Table tbl
    for i in range(len(tables_to_merge)):
        tbl = tables_to_merge[i]
        c_tables_to_merge.push_back(tbl.view())

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
