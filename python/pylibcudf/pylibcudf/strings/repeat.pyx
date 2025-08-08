# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport repeat as cpp_repeat
from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.stream cimport Stream

from ..utils cimport _get_stream

__all__ = ["repeat_strings"]

cpdef Column repeat_strings(
    Column input, ColumnorSizeType repeat_times, Stream stream=None
):
    """
    Repeat each string in the given strings column by the numbers
    of times given in another numeric column.

    For details, see :cpp:func:`cudf::strings::repeat`.

    Parameters
    ----------
    input : Column
        The column containing strings to repeat.
    repeat_times : Column or int
        Number(s) of times that the corresponding input strings
        for each row are repeated.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column containing the repeated strings.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    if ColumnorSizeType is Column:
        with nogil:
            c_result = cpp_repeat.repeat_strings(
                input.view(),
                repeat_times.view(),
                stream.view()
            )
    elif ColumnorSizeType is size_type:
        with nogil:
            c_result = cpp_repeat.repeat_strings(
                input.view(),
                repeat_times,
                stream.view()
            )
    else:
        raise ValueError("repeat_times must be size_type or integer")

    return Column.from_libcudf(move(c_result), stream)
