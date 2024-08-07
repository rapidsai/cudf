# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings cimport strip as cpp_strip
from cudf._lib.pylibcudf.scalar cimport Scalar
from cudf._lib.pylibcudf.strings.side_type cimport side_type


cpdef Column strip(
    Column input,
    side_type side,
    Scalar to_strip
):
    """Removes the specified characters from the beginning
    or end (or both) of each string.

    For details, see :cpp:func:`cudf::strings::strip`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    side : SideType, default SideType.BOTH
        Indicates characters are to be stripped from the beginning,
        end, or both of each string; Default is both
    to_strip : Scalar
        UTF-8 encoded characters to strip from each string;
        Default is empty string which indicates strip whitespace characters

    Returns
    -------
    pylibcudf.Column
        New strings column.
    """

    cdef unique_ptr[column] c_result
    cdef string_scalar* cpp_to_strip
    cpp_to_strip = <string_scalar *>(to_strip.c_obj.get())

    with nogil:
        c_result = cpp_strip.strip(
            input.view(),
            side,
            dereference(cpp_to_strip)
        )

    return Column.from_libcudf(move(c_result))
