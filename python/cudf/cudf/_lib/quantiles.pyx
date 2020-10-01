# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar
from cudf._lib.table cimport Table
from cudf._lib.types cimport (
    underlying_type_t_order,
    underlying_type_t_null_order,
    underlying_type_t_sorted,
    underlying_type_t_interpolation,
)
from cudf._lib.types import Interpolation
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport (
    interpolation,
    null_order,
    order,
    sorted,
    order_info,
)
from cudf._lib.cpp.quantiles cimport (
    quantile as cpp_quantile,
    quantiles as cpp_quantiles,
)


def quantile(
    Column input,
    object q,
    str interp,
    Column ordered_indices,
    bool exact,

):
    cdef column_view c_input = input.view()
    cdef column_view c_ordered_indices = (
        column_view() if ordered_indices is None
        else ordered_indices.view()
    )
    cdef interpolation c_interp = <interpolation>(
        <underlying_type_t_interpolation> Interpolation[interp.upper()]
    )
    cdef bool c_exact = exact

    cdef vector[double] c_q
    c_q.reserve(len(q))

    for value in q:
        c_q.push_back(value)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_quantile(
                c_input,
                c_q,
                c_interp,
                c_ordered_indices,
                c_exact,
            )
        )

    return Column.from_unique_ptr(move(c_result))


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
