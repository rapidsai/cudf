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


# TODO: This should be cpdefed, but cpdefing
# will require creating a Cython mirror for out_of_bounds_policy.
cdef Table gather(
    TableView source_table,
    ColumnView gather_map,
    cpp_copying.out_of_bounds_policy bounds_policy
):
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_copying.gather(
                dereference(source_table.get()),
                dereference(gather_map.get()),
                bounds_policy
            )
        )
    return Table.from_table(move(c_result))
