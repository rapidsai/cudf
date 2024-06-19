# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.strings cimport find as cpp_find
from cudf._lib.pylibcudf.scalar cimport Scalar

from cython.operator import dereference

from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar


cpdef Column find(
    Column input,
    ColumnOrScalar target,
    size_type start=0,
    size_type stop=-1
):
    """Returns a column of character position values where the target string is
    first found in each string of the provided column.

    ``target`` may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`find`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for in each string
    start : size_type
        First character position to include in the search
    stop : size_type
        Last position (exclusive) to include in the search. Default of -1 will
        search to the end of the string.

    Returns
    -------
    pylibcudf.Column
        New integer column with character position values
    """
    cdef unique_ptr[column] result
    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.find(
                    input.view(),
                    target.view(),
                    start
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.find(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get())),
                    start,
                    stop
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))


cpdef Column rfind(
    Column input,
    Scalar target,
    size_type start=0,
    size_type stop=-1
):
    """
    Returns a column of character position values where the target string is
    first found searching from the end of each string.

    For details, see :cpp:func:`rfind`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Scalar
        String to search for in each string
    start : size_type
        First character position to include in the search
    stop : size_type
        Last position (exclusive) to include in the search. Default of -1 will
        search to the end of the string.

    Returns
    -------
    pylibcudf.Column
        New integer column with character position values
    """
    cdef unique_ptr[column] result
    with nogil:
        result = move(
            cpp_find.rfind(
                input.view(),
                dereference(<string_scalar*>(target.c_obj.get())),
                start,
                stop
            )
        )
    return Column.from_libcudf(move(result))


cpdef Column contains(
    Column input,
    ColumnOrScalar target,
):
    """
    Returns a column of boolean values for each string where true indicates the
    corresponding target string was found within that string in the provided
    column.

    ``target`` may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`contains`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for in each string

    Returns
    -------
    pylibcudf.Column
        New boolean column with True for each string that contains the target
    """
    cdef unique_ptr[column] result
    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.contains(
                    input.view(),
                    target.view()
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.contains(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get()))
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))


cpdef Column starts_with(
    Column input,
    ColumnOrScalar target,
):
    """
    Returns a column of boolean values for each string where true indicates the
    target string was found at the beginning of the string in the provided
    column.

    ``target`` may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`starts_with`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for at the beginning of each string

    Returns
    -------
    pylibcudf.Column
        New boolean column with True for each string that starts with the target
    """
    cdef unique_ptr[column] result

    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.starts_with(
                    input.view(),
                    target.view()
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.starts_with(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get()))
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))

cpdef Column ends_with(
    Column input,
    ColumnOrScalar target,
):
    """
    Returns a column of boolean values for each string where true indicates the
    target string was found at the end of the string in the provided column.

    ``target`` may be a
    :py:class:`~cudf._lib.pylibcudf.column.Column` or a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`ends_with`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for at the end of each string

    Returns
    -------
    pylibcudf.Column
        New boolean column with True for each string that ends with the target
    """
    cdef unique_ptr[column] result
    if ColumnOrScalar is Column:
        with nogil:
            result = move(
                cpp_find.ends_with(
                    input.view(),
                    target.view()
                )
            )
    elif ColumnOrScalar is Scalar:
        with nogil:
            result = move(
                cpp_find.ends_with(
                    input.view(),
                    dereference(<string_scalar*>(target.c_obj.get()))
                )
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result))
