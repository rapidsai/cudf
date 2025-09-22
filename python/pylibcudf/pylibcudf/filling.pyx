# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.filling cimport (
    fill as cpp_fill,
    fill_in_place as cpp_fill_in_place,
    repeat as cpp_repeat,
    sequence as cpp_sequence,
    calendrical_month_sequence as cpp_calendrical_month_sequence
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource


__all__ = [
    "fill",
    "fill_in_place",
    "repeat",
    "sequence",
    "calendrical_month_sequence",
]

cpdef Column fill(
    Column destination,
    size_type begin,
    size_type end,
    Scalar value,
    Stream stream=None,
    DeviceMemoryResource mr=None,
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
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        The result of the filling operation
    """

    cdef unique_ptr[column] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_fill(
            destination.view(),
            begin,
            end,
            dereference((<Scalar> value).c_obj),
            stream.view(),
            mr.get_mr()
        )
    return Column.from_libcudf(move(result), stream, mr)

cpdef void fill_in_place(
    Column destination,
    size_type begin,
    size_type end,
    Scalar value,
    Stream stream=None,
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
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    None
    """

    stream = _get_stream(stream)

    with nogil:
        cpp_fill_in_place(
            destination.mutable_view(),
            begin,
            end,
            dereference(value.c_obj),
            stream.view()
        )

cpdef Column sequence(
    size_type size,
    Scalar init,
    Scalar step,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
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
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        The result of the sequence operation
    """

    cdef unique_ptr[column] result
    cdef size_type c_size = size

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_sequence(
            c_size,
            dereference(init.c_obj),
            dereference(step.c_obj),
            stream.view(),
            mr.get_mr()
        )
    return Column.from_libcudf(move(result), stream, mr)


cpdef Table repeat(
    Table input_table,
    ColumnOrSize count,
    Stream stream=None,
    DeviceMemoryResource mr=None,
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
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Table
        The result of the repeat operation
    """

    cdef unique_ptr[table] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    if ColumnOrSize is Column:
        with nogil:
            result = cpp_repeat(
                input_table.view(),
                count.view(),
                stream.view(),
                mr.get_mr()
            )
    if ColumnOrSize is size_type:
        with nogil:
            result = cpp_repeat(
                input_table.view(),
                count,
                stream.view(),
                mr.get_mr()
            )
    return Table.from_libcudf(move(result), stream, mr)


cpdef Column calendrical_month_sequence(
    size_type n,
    Scalar init,
    size_type months,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):

    """Fill destination column from begin to end with value.

    For details, see :cpp:func:`calendrical_month_sequence`.

    Parameters
    ----------
    n : size_type
        Number of timestamps to generate
    init : Scalar
        The initial timestamp
    months : size_type
        Months to increment
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        Timestamps column with sequences of months
    """

    cdef unique_ptr[column] c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_calendrical_month_sequence(
            n,
            dereference(init.c_obj),
            months,
            stream.view(),
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), stream, mr)
