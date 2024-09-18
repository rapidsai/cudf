# Copyright (c) 2024, NVIDIA CORPORATION.
from cython.operator import dereference

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings cimport strip as cpp_strip
from pylibcudf.libcudf.strings.side_type cimport side_type


cpdef Column strip(Column input, side_type side, Scalar to_strip):
    """
    Removes the specified characters from the beginning or end
    (or both) of each string.

    For details, see :cpp:func:`cudf::strings::strip`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    side : SideType
        Indicates characters are to be stripped from the
        beginning, end, or both of each string.
    to_strip : Scalar
        UTF-8 encoded characters to strip from each string.

    Returns
    -------
    Column
        New column with padded strings.
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_to_strip = <const string_scalar*>(
        to_strip.c_obj.get()
    )

    with nogil:
        c_result = move(
            cpp_strip.strip(
                input.view(),
                side,
                dereference(c_to_strip),
            )
        )

    return Column.from_libcudf(move(c_result))
