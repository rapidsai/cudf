# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport find_multiple as cpp_find_multiple
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream
from rmm.pylibrmm.stream cimport Stream

__all__ = ["find_multiple", "contains_multiple"]

cpdef Column find_multiple(Column input, Column targets, Stream stream=None):
    """
    Returns a lists column with character position values where each
    of the target strings are found in each string.

    For details, see :cpp:func:`cudf::strings::find_multiple`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    targets : Column
        Strings to search for in each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Lists column with character position values
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_find_multiple.find_multiple(
            input.view(),
            targets.view(),
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)


cpdef Table contains_multiple(Column input, Column targets, Stream stream=None):
    """
    Returns a table of boolean values where each column indicates
    whether the corresponding target is found at that row.

    For details, see :cpp:func:`cudf::strings::contains_multiple`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    targets : Column
        Strings to search for in each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        Columns of booleans
    """
    cdef unique_ptr[table] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_find_multiple.contains_multiple(
            input.view(),
            targets.view(),
            stream.view()
        )

    return Table.from_libcudf(move(c_result), stream)
