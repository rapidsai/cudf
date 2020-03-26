# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move
from cudf._libxx.types cimport (
    underlying_type_t_order,
    underlying_type_t_null_order,
    underlying_type_t_sorted,
    underlying_type_t_interpolation,
)

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport scalar
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.types cimport (
    interpolation,
    null_order,
    order,
    sorted,
    order_info,
)
from cudf._libxx.cpp.quantiles cimport (
    quantile as cpp_quantile,
    quantiles as cpp_quantiles,
)


def quantile(
    Column input,
    double q,
    object interp,
    object is_sorted,
    object column_order,
    object null_precedence,
    bool exact,

):
    cdef column_view c_input = input.view()
    cdef interpolation c_interp = <interpolation>(
        <underlying_type_t_interpolation> interp
    )
    cdef bool c_is_sorted = column_order.is_sorted
    cdef order_info c_column_order
    cdef bool c_exact = exact

    c_column_order.ordering = <order>(
        <underlying_type_t_order> column_order.ordering
    )
    c_column_order.ordering = <order>(
        <underlying_type_t_order> column_order
    )
    c_column_order.null_ordering = <null_order>(
        <underlying_type_t_null_order> null_precedence
    )

    cdef vector[double] c_q
    c_q.reserve(len(q))

    for value in q:
        c_q.push_back(value)

    cdef unique_ptr[scalar] c_result

    with nogil:
        c_result = move(
            cpp_quantile(
                c_input,
                c_q,
                c_interp,
                c_column_order,
                c_exact,
            )
        )

    return Scalar.from_unique_ptr(move(c_result))


def quantiles(Table source_table,
              vector[double] q,
              object interp,
              object is_input_sorted,
              list column_order,
              list null_precedence):
    cdef table_view c_input = source_table.data_view()
    cdef vector[double] c_q = q
    cdef interpolation c_interp = <interpolation>(
        <underlying_type_t_interpolation> interp
    )
    cdef sorted c_is_input_sorted = <sorted>(
        <underlying_type_t_sorted> is_input_sorted
    )
    cdef vector[order] c_column_order
    cdef vector[null_order] c_null_precedence

    c_column_order.reserve(len(column_order))
    c_null_precedence.reserve(len(null_precedence))

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
            cpp_quantiles(
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
