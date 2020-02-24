# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
cimport cudf._libxx.includes.quantiles as cpp_quantiles

def quantiles(Table source_table,
              vector[double] q,
              object interp,
              object is_input_sorted,
              list column_order,
              list null_precedence):
    cdef table_view c_input = source_table.view()
    cdef vector[double] c_q = q
    cdef interpolation c_interp = <interpolation>(<underlying_type_t_interpolation> interp)
    cdef sorted c_is_input_sorted = <sorted>(<underlying_type_t_sorted> is_input_sorted)
    cdef vector[order] c_column_order = vector[order](len(column_order))
    cdef vector[null_order] c_null_precedence = vector[null_order](len(null_precedence))

    for value in column_order:
        c_column_order.push_back(
            <order>(<underlying_type_t_order> value)
        )

    for value in null_precedence:
        c_null_precedence.push_back(
            <null_order>(<underlying_type_t_null_order> value)
        )

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_quantiles.quantiles(
                c_input,
                c_q,
                c_interp,
                c_is_input_sorted,
                c_column_order,
                c_null_precedence
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=source_table._index_names
    )
