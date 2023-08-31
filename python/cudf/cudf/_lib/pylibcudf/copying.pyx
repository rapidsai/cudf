# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

# TODO: We want to make cpp a more full-featured package so that we can access
# directly from that. It will make namespacing much cleaner in pylibcudf. What
# we really want here would be
# cimport libcudf... libcudf.copying.algo(...)
from cudf._lib.cpp cimport copying as cpp_copying
from cudf._lib.cpp.copying cimport out_of_bounds_policy
from cudf._lib.cpp.libcpp.functional cimport reference_wrapper

from cudf._lib.cpp.copying import \
    out_of_bounds_policy as OutOfBoundsPolicy  # no-cython-lint

from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table

from .column cimport Column
from .scalar cimport Scalar
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


ctypedef const scalar constscalar

cpdef Table scatter (
    list source_scalars,
    Column indices,
    Table target,
):
    """Scatter source_scalars into target according to the indices.

    For details on the implementation, see cudf::scatter in libcudf.

    Parameters
    ----------
    source_scalars : List[Scalar]
        A list containing one scalar for each column in target.
    indices : Column
        The rows of the target into which the source_scalars should be written.
    target : Table
        The table into which data should be written.

    Returns
    -------
    pylibcudf.Table
        The result of the scatter
    """
    cdef unique_ptr[table] c_result
    # TODO: This doesn't require the constscalar ctypedef
    cdef vector[reference_wrapper[const scalar]] c_scalars
    c_scalars.reserve(len(source_scalars))
    cdef Scalar d_slr
    for d_slr in source_scalars:
        c_scalars.push_back(
            # TODO: This requires the constscalar ctypedef
            # Possibly the same as https://github.com/cython/cython/issues/4180
            reference_wrapper[constscalar](d_slr.get()[0])
        )

    with nogil:
        c_result = move(
            cpp_copying.scatter(
                c_scalars,
                indices.view(),
                target.view(),
            )
        )

    return Table.from_libcudf(move(c_result))
