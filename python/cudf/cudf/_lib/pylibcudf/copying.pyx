# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

# TODO: We want to make cpp a more full-featured package so that we can access
# directly from that. It will make namespacing much cleaner in pylibcudf. What
# we really want here would be
# cimport libcudf... libcudf.copying.algo(...)
from cudf._lib.cpp cimport copying as cpp_copying
from cudf._lib.cpp.copying cimport out_of_bounds_policy

from cudf._lib.cpp.copying import \
    out_of_bounds_policy as OutOfBoundsPolicy  # no-cython-lint

from cudf._lib.cpp.table.table cimport table

from .column cimport Column
from .table cimport Table


# TODO: Is it OK to reference the corresponding libcudf algorithm in the
# documentation? Otherwise there's a lot of room for duplication.
cpdef Table gather(
    Table source_table,
    Column gather_map,
    out_of_bounds_policy bounds_policy
):
    """Select rows from source_table according to the provided gather_map.

    For details on the implementation, see cudf::gather in libcudf.

    Parameters
    ----------
    source_table : Table
        The table object from which to pull data.
    gather_map : Column
        The list of row indices to pull out of the source table.
    bounds_policy : out_of_bounds_policy
        Controls whether out of bounds indices are checked and nullified in the
        output or if indices are assumed to be in bounds.

    Returns
    -------
    pylibcudf.Table
        The result of the gather
    """
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_copying.gather(
                source_table.view(),
                gather_map.view(),
                bounds_policy
            )
        )
    return Table.from_libcudf(move(c_result))
