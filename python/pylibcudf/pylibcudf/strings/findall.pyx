# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport findall as cpp_findall
from pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column findall(Column input, RegexProgram prog):
    """
    Returns a lists column of strings for each matching occurrence using
    the regex_program pattern within each string.

    For details, see For details, see :cpp:func:`cudf::strings::findall`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    Column
        New lists column of strings
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_findall.findall(
                input.view(),
                prog.c_obj.get()[0]
            )
        )

    return Column.from_libcudf(move(c_result))
