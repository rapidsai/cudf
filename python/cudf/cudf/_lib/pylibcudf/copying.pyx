# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

# TODO: We want to make cpp a more full-featured package so that we can access
# directly from that. It will make namespacing much cleaner in pylibcudf. What
# we really want here would be
# cimport libcudf... libcudf.copying.algo(...)
from cudf._lib.cpp cimport copying as cpp_copying
from cudf._lib.cpp.table.table cimport table

from .column_view cimport ColumnView
from .table cimport Table
from .table_view cimport TableView


# Cython doesn't support scoped enumerations. It assumes that enums correspond
# to their underlying value types and will thus attempt operations that are
# invalid. This code will ensure that these values are explicitly cast to the
# underlying type before casting to the final type.
cdef cpp_copying.out_of_bounds_policy py_policy_to_c_policy(
    OutOfBoundsPolicy py_policy
) nogil:
    return <cpp_copying.out_of_bounds_policy> (
        <underlying_type_t_out_of_bounds_policy> py_policy
    )


cpdef Table gather(
    TableView source_table,
    ColumnView gather_map,
    OutOfBoundsPolicy bounds_policy
):
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_copying.gather(
                dereference(source_table.get()),
                dereference(gather_map.get()),
                py_policy_to_c_policy(bounds_policy)
            )
        )
    return Table.from_table(move(c_result))
