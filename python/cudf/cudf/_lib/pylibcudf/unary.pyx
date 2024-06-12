# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf cimport unary as cpp_unary
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.unary cimport unary_operator

from cudf._lib.pylibcudf.libcudf.unary import \
    unary_operator as UnaryOperator  # no-cython-lint

from .column cimport Column
from .types cimport DataType


cpdef Column unary_operation(Column input, unary_operator op):
    """Perform a unary operation on a column.

    For details, see :cpp:func:`unary_operation`.

    Parameters
    ----------
    input : Column
        The column to operate on.
    op : UnaryOperator
        The operation to perform.

    Returns
    -------
    pylibcudf.Column
        The result of the unary operation
    """
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.unary_operation(input.view(), op))

    return Column.from_libcudf(move(result))


cpdef Column is_null(Column input):
    """Check whether elements of a column are null.

    For details, see :cpp:func:`is_null`.

    Parameters
    ----------
    input : Column
        The column to check.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing null values.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_null(input.view()))

    return Column.from_libcudf(move(result))


cpdef Column is_valid(Column input):
    """Check whether elements of a column are valid.

    For details, see :cpp:func:`is_valid`.

    Parameters
    ----------
    input : Column
        The column to check.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing valid values.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_valid(input.view()))

    return Column.from_libcudf(move(result))


cpdef Column cast(Column input, DataType data_type):
    """Cast a column to a different data type.

    For details, see :cpp:func:`cast`.

    Parameters
    ----------
    input : Column
        The column to check.
    data_type : DataType
        The data type to cast to.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing null values.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.cast(input.view(), data_type.c_obj))

    return Column.from_libcudf(move(result))


cpdef Column is_nan(Column input):
    """Check whether elements of a column are nan.

    For details, see :cpp:func:`is_nan`.

    Parameters
    ----------
    input : Column
        The column to check.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing nan values.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_nan(input.view()))

    return Column.from_libcudf(move(result))


cpdef Column is_not_nan(Column input):
    """Check whether elements of a column are not nan.

    For details, see :cpp:func:`is_not_nan`.

    Parameters
    ----------
    input : Column
        The column to check.

    Returns
    -------
    pylibcudf.Column
        A boolean column with ``True`` representing non-nan values.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = move(cpp_unary.is_not_nan(input.view()))

    return Column.from_libcudf(move(result))
