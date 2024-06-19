# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.filling cimport (
    fill as cpp_fill,
    fill_in_place as cpp_fill_in_place,
    repeat as cpp_repeat,
    sequence as cpp_sequence,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table


cpdef Column fill(
    Column destination,
    size_type begin,
    size_type end,
    Scalar value,
):

    """Fill destination column from begin to end with value.

    For details, see :cpp:func:`fill`.

    Parameters
    ----------
    destination : Column
        The column to be filled
    begin : size_type
        The index to begin filling from.
    end : size_type
        The index at which to stop filling.
    value : Scalar
        The value to fill with.

    Returns
    -------
    pylibcudf.Column
        The result of the filling operation
    """

    cdef unique_ptr[column] result
    with nogil:
        result = move(
            cpp_fill(
                destination.view(),
                begin,
                end,
                dereference((<Scalar> value).c_obj)
            )
        )
    return Column.from_libcudf(move(result))

cpdef void fill_in_place(
    Column destination,
    size_type begin,
    size_type end,
    Scalar value,
):

    """Fill destination column in place from begin to end with value.

    For details, see :cpp:func:`fill_in_place`.

    Parameters
    ----------
    destination : Column
        The column to be filled
    begin : size_type
        The index to begin filling from.
    end : size_type
        The index at which to stop filling.
    value : Scalar
        The value to fill with.
    """

    with nogil:
        cpp_fill_in_place(
            destination.mutable_view(),
            begin,
            end,
            dereference(value.c_obj)
        )

cpdef Column sequence(size_type size, Scalar init, Scalar step):
    """Create a sequence column of size ``size`` with initial value ``init`` and step
    ``step``.

    For details, see :cpp:func:`sequence`.

    Parameters
    ----------
    size : int
        The size of the sequence
    init : Scalar
        The initial value of the sequence
    step : Scalar
        The step of the sequence
    Returns
    -------
    pylibcudf.Column
        The result of the sequence operation
    """

    cdef unique_ptr[column] result
    cdef size_type c_size = size
    with nogil:
        result = move(
            cpp_sequence(
                c_size,
                dereference(init.c_obj),
                dereference(step.c_obj),
            )
        )
    return Column.from_libcudf(move(result))


cpdef Table repeat(
    Table input_table,
    ColumnOrSize count
):
    """Repeat rows of a Table.

    If an integral value is specified for ``count``, every row is repeated ``count``
    times. If ``count`` is a column, the number of repetitions of each row is defined
    by the value at the corresponding index of ``count``.

    For details, see :cpp:func:`repeat`.

    Parameters
    ----------
    input_table : Table
        The table to be repeated
    count : Union[Column, size_type]
        Integer value to repeat each row by or
        non-nullable column of an integral type

    Returns
    -------
    pylibcudf.Table
        The result of the repeat operation
    """

    cdef unique_ptr[table] result

    if ColumnOrSize is Column:
        with nogil:
            result = move(
                cpp_repeat(
                    input_table.view(),
                    count.view()
                )
            )
    if ColumnOrSize is size_type:
        with nogil:
            result = move(
                cpp_repeat(
                    input_table.view(),
                    count
                )
            )
    return Table.from_libcudf(move(result))
