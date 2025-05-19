# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport find_multiple as cpp_find_multiple

__all__ = ["find_multiple"]

cpdef Column find_multiple(Column input, Column targets):
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

    Returns
    -------
    Column
        Lists column with character position values
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_find_multiple.find_multiple(
            input.view(),
            targets.view()
        )

    return Column.from_libcudf(move(c_result))
