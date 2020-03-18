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
              Column sortmap,
              bool cast_to_doubles):
    cdef table_view c_input = source_table.data_view()
    cdef vector[double] c_q = q
    cdef libcudf_types.interpolation c_interp = <libcudf_types.interpolation>(
        <underlying_type_t_interpolation> interp
    )
    cdef column_view c_sortmap

    if sortmap is None:
        c_sortmap = column_view()
    else:
        c_sortmap = sortmap.view()

    cdef bool c_cast_to_doubles = cast_to_doubles

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_quantiles.quantiles(
                c_input,
                c_q,
                c_interp,
                c_sortmap,
                c_cast_to_doubles
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names
    )
