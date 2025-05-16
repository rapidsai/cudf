# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport repeat as cpp_repeat
from pylibcudf.libcudf.types cimport size_type

__all__ = ["repeat_strings"]

cpdef Column repeat_strings(Column input, ColumnorSizeType repeat_times):
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

    Returns
    -------
    Column
        New column containing the repeated strings.
    """
    cdef unique_ptr[column] c_result

    if ColumnorSizeType is Column:
        with nogil:
            c_result = cpp_repeat.repeat_strings(
                input.view(),
                repeat_times.view()
            )
    elif ColumnorSizeType is size_type:
        with nogil:
            c_result = cpp_repeat.repeat_strings(
                input.view(),
                repeat_times
            )
    else:
        raise ValueError("repeat_times must be size_type or integer")

    return Column.from_libcudf(move(c_result))
