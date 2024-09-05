# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport contains as cpp_contains
from pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column contains_re(
    Column input,
    RegexProgram prog
):
    """Returns a boolean column identifying rows which match the given
    regex_program object.

    For details, see :cpp:func:`cudf::strings::contains_re`.

    Parameters
    ----------
    input : Column
        The input strings
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    pylibcudf.Column
        New column of boolean results for each string
    """

    cdef unique_ptr[column] result

    with nogil:
        result = cpp_contains.contains_re(
            input.view(),
            prog.c_obj.get()[0]
        )

    return Column.from_libcudf(move(result))
