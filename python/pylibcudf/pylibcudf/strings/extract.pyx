# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport extract as cpp_extract
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table
from pylibcudf.libcudf.types cimport size_type

__all__ = ["extract", "extract_all_record", "extract_single"]

cpdef Table extract(Column input, RegexProgram prog):
    """
    Returns a table of strings columns where each column
    corresponds to the matching group specified in the given
    egex_program object.

    For details, see :cpp:func:`cudf::strings::extract`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    Table
        Columns of strings extracted from the input column.
    """
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = cpp_extract.extract(
            input.view(),
            prog.c_obj.get()[0]
        )

    return Table.from_libcudf(move(c_result))


cpdef Column extract_all_record(Column input, RegexProgram prog):
    """
    Returns a lists column of strings where each string column
    row corresponds to the matching group specified in the given
    regex_program object.

    For details, see :cpp:func:`cudf::strings::extract_all_record`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    Column
        Lists column containing strings extracted from the input column
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_extract.extract_all_record(
            input.view(),
            prog.c_obj.get()[0]
        )

    return Column.from_libcudf(move(c_result))


cpdef Column extract_single(
    Column input,
    RegexProgram prog,
    size_type group
):
    """
    Returns a column of strings where each string corresponds to the
    matching group specified in the given regex_program object.

    For details, see :cpp:func:`cudf::strings::extract_single`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    prog : RegexProgram
        Regex program instance
    group : size_type
        Index of the group number to extract

    Returns
    -------
    Column
        Column of strings extracted from the input column
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_extract.extract_single(
            input.view(),
            prog.c_obj.get()[0],
            group
        )

    return Column.from_libcudf(move(c_result))
