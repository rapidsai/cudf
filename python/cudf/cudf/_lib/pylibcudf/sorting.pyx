# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp cimport sorting as cpp_sorting
from cudf._lib.cpp.aggregation cimport rank_method
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport null_order, null_policy, order

from .column cimport Column
from .table cimport Table


cpdef Column sorted_order(Table source_table, list column_order, list null_precedence):
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.sorted_order(
                source_table.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Column.from_libcudf(move(c_result))


cpdef Column stable_sorted_order(
    Table source_table,
    list column_order,
    list null_precedence,
):
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.stable_sorted_order(
                source_table.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Column.from_libcudf(move(c_result))


cpdef Column rank(
    Column input_view,
    rank_method method,
    order column_order,
    null_policy null_handling,
    null_order null_precedence,
    bool percentage,
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_sorting.rank(
                input_view.view(),
                method,
                column_order,
                null_handling,
                null_precedence,
                percentage,
            )
        )
    return Column.from_libcudf(move(c_result))


cpdef bool is_sorted(Table tbl, list column_order, list null_precedence):
    cdef bool c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.is_sorted(
                tbl.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return c_result


cpdef Table segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
):
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.segmented_sort_by_key(
                values.view(),
                keys.view(),
                segment_offsets.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table stable_segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
):
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.stable_segmented_sort_by_key(
                values.view(),
                keys.view(),
                segment_offsets.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
):
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.sort_by_key(
                values.view(),
                keys.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table stable_sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
):
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.stable_sort_by_key(
                values.view(),
                keys.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table sort(Table source_table, list column_order, list null_precedence):
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = move(
            cpp_sorting.sort(
                source_table.view(),
                c_orders,
                c_null_precedence,
            )
        )
    return Table.from_libcudf(move(c_result))
