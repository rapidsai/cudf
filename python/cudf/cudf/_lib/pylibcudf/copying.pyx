# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

# TODO: We want to make cpp a more full-featured package so that we can access
# directly from that. It will make namespacing much cleaner in pylibcudf. What
# we really want here would be
# cimport libcudf... libcudf.copying.algo(...)
from cudf._lib.cpp cimport copying as cpp_copying
from cudf._lib.cpp.table.table cimport table

from .column cimport Column
from .table cimport Table


cdef inline cpp_copying.out_of_bounds_policy py_policy_to_c_policy(
    OutOfBoundsPolicy py_policy
) nogil:
    return <cpp_copying.out_of_bounds_policy> (
        <underlying_type_t_out_of_bounds_policy> py_policy
    )


cpdef Table gather(
    Table source_table,
    Column gather_map,
    OutOfBoundsPolicy bounds_policy
):
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_copying.gather(
                source_table.view(),
                gather_map.view(),
                py_policy_to_c_policy(bounds_policy)
            )
        )
    return Table.from_libcudf(move(c_result))
