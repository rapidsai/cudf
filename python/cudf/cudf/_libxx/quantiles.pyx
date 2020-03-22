# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
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
from cudf._libxx.cpp.column.column_view cimport column_view
cimport cudf._libxx.cpp.types as libcudf_types
cimport cudf._libxx.cpp.quantiles as cpp_quantiles


def quantiles(Table source_table,
              vector[double] q,
              object interp,
              Column ordered_indices,
              bool retain_dtype):
    cdef table_view c_input = source_table.data_view()
    cdef vector[double] c_q = q
    cdef libcudf_types.interpolation c_interp = <libcudf_types.interpolation>(
        <underlying_type_t_interpolation> interp
    )
    cdef column_view c_ordered_indices

    if ordered_indices is None:
        c_ordered_indices = column_view()
    else:
        c_ordered_indices = ordered_indices.view()

    cdef bool c_retain_types = retain_dtype

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_quantiles.quantiles(
                c_input,
                c_q,
                c_interp,
                c_ordered_indices,
                c_retain_types
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names
    )
