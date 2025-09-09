# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport wrap as cpp_wrap
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream
from rmm.pylibrmm.stream cimport Stream

__all__ = ["wrap"]

cpdef Column wrap(Column input, size_type width, Stream stream=None):
    """
    Wraps strings onto multiple lines shorter than `width` by
    replacing appropriate white space with
    new-line characters (ASCII 0x0A).

    For details, see :cpp:func:`cudf::strings::wrap`.

    Parameters
    ----------
    input : Column
        String column

    width : int
        Maximum character width of a line within each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of wrapped strings
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_wrap.wrap(
            input.view(),
            width,
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)
