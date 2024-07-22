# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator import dereference

from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport device_buffer

from cudf._lib.pylibcudf.libcudf cimport join as cpp_join
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport null_equality

from .column cimport Column
from .table cimport Table


cdef Column _column_from_gather_map(cpp_join.gather_map_type gather_map):
    # helper to convert a gather map to a Column
    return Column.from_libcudf(
        move(
            make_unique[column](
                move(dereference(gather_map.get())),
                device_buffer(),
                0
            )
        )
    )


cpdef tuple inner_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
):
    """Perform an inner join between two tables.

    For details, see :cpp:func:`inner_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    with nogil:
        c_result = cpp_join.inner_join(left_keys.view(), right_keys.view(), nulls_equal)
    return (
        _column_from_gather_map(move(c_result.first)),
        _column_from_gather_map(move(c_result.second)),
    )


cpdef tuple left_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
):
    """Perform a left join between two tables.

    For details, see :cpp:func:`left_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    with nogil:
        c_result = cpp_join.left_join(left_keys.view(), right_keys.view(), nulls_equal)
    return (
        _column_from_gather_map(move(c_result.first)),
        _column_from_gather_map(move(c_result.second)),
    )


cpdef tuple full_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
):
    """Perform a full join between two tables.

    For details, see :cpp:func:`full_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    with nogil:
        c_result = cpp_join.full_join(left_keys.view(), right_keys.view(), nulls_equal)
    return (
        _column_from_gather_map(move(c_result.first)),
        _column_from_gather_map(move(c_result.second)),
    )


cpdef Column left_semi_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
):
    """Perform a left semi join between two tables.

    For details, see :cpp:func:`left_semi_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result
    with nogil:
        c_result = cpp_join.left_semi_join(
            left_keys.view(),
            right_keys.view(),
            nulls_equal
        )
    return _column_from_gather_map(move(c_result))


cpdef Column left_anti_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal
):
    """Perform a left anti join between two tables.

    For details, see :cpp:func:`left_anti_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result
    with nogil:
        c_result = cpp_join.left_anti_join(
            left_keys.view(),
            right_keys.view(),
            nulls_equal
        )
    return _column_from_gather_map(move(c_result))


cpdef Table cross_join(Table left, Table right):
    """Perform a cross join on two tables.

    For details see :cpp:func:`cross_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right: Table
        The right table to join.

    Returns
    -------
    Table
        The result of cross joining the two inputs.
    """
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_join.cross_join(left.view(), right.view()))
    return Table.from_libcudf(move(result))
