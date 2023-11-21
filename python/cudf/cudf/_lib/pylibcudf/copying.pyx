# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator import dereference

from libcpp.functional cimport reference_wrapper
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

# TODO: We want to make cpp a more full-featured package so that we can access
# directly from that. It will make namespacing much cleaner in pylibcudf. What
# we really want here would be
# cimport libcudf... libcudf.copying.algo(...)
from cudf._lib.cpp cimport copying as cpp_copying
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.copying cimport mask_allocation_policy, out_of_bounds_policy
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type

from cudf._lib.cpp.copying import \
    mask_allocation_policy as MaskAllocationPolicy  # no-cython-lint
from cudf._lib.cpp.copying import \
    out_of_bounds_policy as OutOfBoundsPolicy  # no-cython-lint

from .column cimport Column
from .table cimport Table

# workaround for https://github.com/cython/cython/issues/3885
ctypedef const scalar constscalar


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


cpdef Column shift(Column input, size_type offset, Scalar fill_values):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_copying.shift(
                input.view(),
                offset,
                dereference(fill_values.c_obj)
            )
        )
    return Column.from_libcudf(move(c_result))


cpdef Table scatter(object source, Column scatter_map, Table target_table):
    cdef unique_ptr[table] c_result
    cdef vector[reference_wrapper[constscalar]] source_scalars
    cdef Scalar slr

    if isinstance(source, Table):
        with nogil:
            c_result = move(
                cpp_copying.scatter(
                    (<Table> source).view(),
                    scatter_map.view(),
                    target_table.view(),
                )
            )
    elif isinstance(source, list):  # TODO: is list too restrictive?
        for slr in source:
            source_scalars.push_back(
                reference_wrapper[constscalar](dereference(slr.c_obj))
            )

        with nogil:
            c_result = move(
                cpp_copying.scatter(
                    source_scalars,
                    scatter_map.view(),
                    target_table.view(),
                )
            )
    else:
        raise ValueError("source must be a Table or list[Scalar]")

    return Table.from_libcudf(move(c_result))


cpdef object empty_like(object input):
    cdef unique_ptr[column] c_column_result
    cdef unique_ptr[table] c_table_result
    if isinstance(input, Column):
        with nogil:
            c_column_result = move(
                cpp_copying.empty_like(
                    (<Column> input).view(),
                )
            )
        return Column.from_libcudf(move(c_column_result))
    elif isinstance(input, Table):
        with nogil:
            c_table_result = move(
                cpp_copying.empty_like(
                    (<Table> input).view(),
                )
            )
        return Table.from_libcudf(move(c_table_result))
    else:
        raise ValueError("input must be a Table or a Column")


cpdef Column allocate_like(
    Column input_column, mask_allocation_policy policy, size=None
):
    cdef unique_ptr[column] c_result
    cdef size_type c_size = size if size is not None else input_column.size()

    with nogil:
        c_result = move(
            cpp_copying.allocate_like(
                input_column.view(),
                c_size,
                policy,
            )
        )

    return Column.from_libcudf(move(c_result))


cpdef Column copy_if_else(object lhs, object rhs, Column boolean_mask):
    cdef unique_ptr[column] result

    if isinstance(lhs, Column) and isinstance(rhs, Column):
        with nogil:
            result = move(
                cpp_copying.copy_if_else(
                    (<Column> lhs).view(),
                    (<Column> rhs).view(),
                    boolean_mask.view()
                )
            )
    elif isinstance(lhs, Column) and isinstance(rhs, Scalar):
        with nogil:
            result = move(
                cpp_copying.copy_if_else(
                    (<Column> lhs).view(),
                    dereference((<Scalar> rhs).c_obj),
                    boolean_mask.view()
                )
            )
    elif isinstance(lhs, Scalar) and isinstance(rhs, Column):
        with nogil:
            result = move(
                cpp_copying.copy_if_else(
                    dereference((<Scalar> lhs).c_obj),
                    (<Column> rhs).view(),
                    boolean_mask.view()
                )
            )
    elif isinstance(lhs, Scalar) and isinstance(rhs, Scalar):
        with nogil:
            result = move(
                cpp_copying.copy_if_else(
                    dereference((<Scalar> lhs).c_obj),
                    dereference((<Scalar> rhs).c_obj),
                    boolean_mask.view()
                )
            )
    else:
        raise ValueError(f"Invalid arguments {lhs} and {rhs}")

    return Column.from_libcudf(move(result))


cpdef Table boolean_mask_scatter(object input, Table target, Column boolean_mask):
    cdef unique_ptr[table] result
    cdef vector[reference_wrapper[const scalar]] c_scalars
    cdef Scalar slr

    # TODO: Could generalize to sequence
    if isinstance(input, list):
        if not isinstance(input[0], Scalar):
            raise TypeError("input must be a list of scalars")

        c_scalars.reserve(len(input))
        for slr in input:
            c_scalars.push_back(
                # TODO: This requires the constscalar ctypedef
                # https://github.com/cython/cython/issues/4180
                reference_wrapper[constscalar](dereference(slr.c_obj))
            )

        with nogil:
            result = move(
                cpp_copying.boolean_mask_scatter(
                    c_scalars,
                    target.view(),
                    boolean_mask.view(),
                )
            )
    elif isinstance(input, Table):
        with nogil:
            result = move(
                cpp_copying.boolean_mask_scatter(
                    (<Table> input).view(),
                    target.view(),
                    boolean_mask.view()
                )
            )
    else:
        raise ValueError(f"Invalid argument {input}")

    return Table.from_libcudf(move(result))
