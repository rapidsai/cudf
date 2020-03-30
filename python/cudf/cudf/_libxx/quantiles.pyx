# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move
from cudf._libxx.types cimport (
    underlying_type_t_order,
    underlying_type_t_null_order,
    underlying_type_t_sorted,
    underlying_type_t_interpolation
)

from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
cimport cudf._libxx.cpp.types as libcudf_types
cimport cudf._libxx.cpp.quantiles as cpp_quantiles


def quantiles(Table source_table,
              vector[double] q,
              object interp,
              object is_input_sorted,
              list column_order,
              list null_precedence):
    cdef table_view c_input = source_table.data_view()
    cdef vector[double] c_q = q
    cdef libcudf_types.interpolation c_interp = <libcudf_types.interpolation>(
        <underlying_type_t_interpolation> interp
    )
    cdef libcudf_types.sorted c_is_input_sorted = <libcudf_types.sorted>(
        <underlying_type_t_sorted> is_input_sorted
    )
    cdef vector[libcudf_types.order] c_column_order
    cdef vector[libcudf_types.null_order] c_null_precedence

    c_column_order.reserve(len(column_order))
    c_null_precedence.reserve(len(null_precedence))

    for value in column_order:
        c_column_order.push_back(
            <libcudf_types.order>(<underlying_type_t_order> value)
        )

    for value in null_precedence:
        c_null_precedence.push_back(
            <libcudf_types.null_order>(<underlying_type_t_null_order> value)
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
        column_names=source_table._column_names
    )
