# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType

from cudf._lib.cpp.filling cimport (
    fill as cpp_fill,
    fill_in_place as cpp_fill_in_place,
)


cpdef Column fill(
    object destination,
    size_type begin,
    size_type end,
    object value,
):

    """Fill destination column from begin to end with value.
    ``destination ``must be a
    :py:class:`~cudf._lib.pylibcudf.column.Column`. ``value`` must be a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.
    For details, see :cpp:func:`fill`.
    Parameters
    ----------
    destination : Column
        The column to be filled
    begin : size_type
        The index to begin filling from.
    end : size_type
        The index at which to stop filling.
    Returns
    -------
    pylibcudf.Column
        The result of the filling operation
    """

    cdef unique_ptr[column] result
    with nogil:
        result = move(
                cpp_fill(
                (<Column> destination).view(),
                begin,
                end,
                dereference((<Scalar> value).c_obj)
            )
        )
    return Column.from_libcudf(move(result))

cpdef void fill_in_place(
    object destination,
    size_type begin,
    size_type end,
    object value,
):

    """Fill destination column in place from begin to end with value.
    ``destination ``must be a
    :py:class:`~cudf._lib.pylibcudf.column.Column`. ``value`` must be a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.
    For details, see :cpp:func:`fill_in_place`.
    Parameters
    ----------
    destination : Column
        The column to be filled
    begin : size_type
        The index to begin filling from.
    end : size_type
        The index at which to stop filling.
    """

    with nogil:
        cpp_fill_in_place(
            (<Column> destination).mutable_view(),
            begin,
            end,
            dereference((<Scalar> value).c_obj)
        )
