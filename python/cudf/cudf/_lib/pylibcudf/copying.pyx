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
from cudf._lib.cpp.column.column_view cimport column_view, mutable_column_view
from cudf._lib.cpp.copying cimport mask_allocation_policy, out_of_bounds_policy
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type

from cudf._lib.cpp.copying import \
    mask_allocation_policy as MaskAllocationPolicy  # no-cython-lint
from cudf._lib.cpp.copying import \
    out_of_bounds_policy as OutOfBoundsPolicy  # no-cython-lint

from .column cimport Column
from .table cimport Table

# This is a workaround for
# https://github.com/cython/cython/issues/4180
# when creating reference_wrapper[constscalar] in the constructor
ctypedef const scalar constscalar


cdef vector[reference_wrapper[const scalar]] _as_vector(list source):
    """Make a vector of reference_wrapper[const scalar] from a list of scalars."""
    cdef vector[reference_wrapper[const scalar]] c_scalars
    c_scalars.reserve(len(source))
    cdef Scalar slr
    for slr in source:
        c_scalars.push_back(
            reference_wrapper[constscalar](dereference((<Scalar?>slr).c_obj)))
    return c_scalars


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


cpdef Table scatter_table(Table source, Column scatter_map, Table target_table):
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_copying.scatter(
                source.view(),
                scatter_map.view(),
                target_table.view(),
            )
        )

    return Table.from_libcudf(move(c_result))


# TODO: Could generalize list to sequence
cpdef Table scatter_scalars(list source, Column scatter_map, Table target_table):
    cdef vector[reference_wrapper[const scalar]] source_scalars = \
        _as_vector(source)

    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_copying.scatter(
                source_scalars,
                scatter_map.view(),
                target_table.view(),
            )
        )

    return Table.from_libcudf(move(c_result))


cpdef object empty_column_like(Column input):
    cdef unique_ptr[column] c_column_result
    with nogil:
        c_column_result = move(
            cpp_copying.empty_like(
                (<Column> input).view(),
            )
        )
    return Column.from_libcudf(move(c_column_result))


cpdef object empty_table_like(Table input):
    cdef unique_ptr[table] c_table_result
    with nogil:
        c_table_result = move(
            cpp_copying.empty_like(
                (<Table> input).view(),
            )
        )
    return Table.from_libcudf(move(c_table_result))


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


cpdef Column copy_range_in_place(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
):
    # Need to initialize this outside the function call so that Cython doesn't
    # try and pass a temporary that decays to an rvalue reference in where the
    # function requires an lvalue reference.
    cdef mutable_column_view target_view = target_column.mutable_view()
    with nogil:
        cpp_copying.copy_range_in_place(
            input_column.view(),
            target_view,
            input_begin,
            input_end,
            target_begin
        )


cpdef Column copy_range(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
):
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_copying.copy_range(
            input_column.view(),
            target_column.view(),
            input_begin,
            input_end,
            target_begin)
        )

    return Column.from_libcudf(move(c_result))


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


cpdef list column_split(Column input_column, list splits):
    cdef vector[size_type] c_splits
    cdef int split
    for split in splits:
        c_splits.push_back(split)

    cdef vector[column_view] c_result
    with nogil:
        c_result = move(
            cpp_copying.split(
                input_column.view(),
                c_splits
            )
        )

    cdef int i
    return [
        Column.from_column_view(c_result[i], input_column)
        for i in range(c_result.size())
    ]


cpdef list table_split(Table input_table, list splits):
    cdef vector[size_type] c_splits = splits
    cdef vector[table_view] c_result
    with nogil:
        c_result = move(
            cpp_copying.split(
                input_table.view(),
                c_splits
            )
        )

    cdef int i
    return [
        Table.from_table_view(c_result[i], input_table)
        for i in range(c_result.size())
    ]


cpdef list column_slice(Column input_column, list indices):
    cdef vector[size_type] c_indices = indices
    cdef vector[column_view] c_result
    with nogil:
        c_result = move(
            cpp_copying.slice(
                input_column.view(),
                c_indices
            )
        )

    cdef int i
    return [
        Column.from_column_view(c_result[i], input_column)
        for i in range(c_result.size())
    ]


cpdef list table_slice(Table input_table, list indices):
    cdef vector[size_type] c_indices = indices
    cdef vector[table_view] c_result
    with nogil:
        c_result = move(
            cpp_copying.slice(
                input_table.view(),
                c_indices
            )
        )

    cdef int i
    return [
        Table.from_table_view(c_result[i], input_table)
        for i in range(c_result.size())
    ]


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


cpdef Table boolean_mask_table_scatter(Table input, Table target, Column boolean_mask):
    cdef unique_ptr[table] result

    with nogil:
        result = move(
            cpp_copying.boolean_mask_scatter(
                (<Table> input).view(),
                target.view(),
                boolean_mask.view()
            )
        )

    return Table.from_libcudf(move(result))


# TODO: Could generalize list to sequence
cpdef Table boolean_mask_scalars_scatter(list input, Table target, Column boolean_mask):
    cdef vector[reference_wrapper[const scalar]] source_scalars = _as_vector(input)

    cdef unique_ptr[table] result
    with nogil:
        result = move(
            cpp_copying.boolean_mask_scatter(
                source_scalars,
                target.view(),
                boolean_mask.view(),
            )
        )

    return Table.from_libcudf(move(result))

cpdef Scalar get_element(Column input_column, size_type index):
    cdef unique_ptr[scalar] c_output
    with nogil:
        c_output = move(
            cpp_copying.get_element(input_column.view(), index)
        )

    return Scalar.from_libcudf(move(c_output))
