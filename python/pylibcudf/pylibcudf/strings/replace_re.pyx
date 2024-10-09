# Copyright (c) 2024, NVIDIA CORPORATION.
from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings cimport replace_re as cpp_replace_re
from pylibcudf.libcudf.strings.regex_program cimport regex_program
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_flags cimport regex_flags
from pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column replace_re(
    Column input,
    Patterns patterns,
    Replacement replacement,
    size_type max_replace_count=-1,
    regex_flags flags=regex_flags.DEFAULT,
):
    """
    For each string, replaces any character sequence matching the given patterns
    with the provided replacement.

    For details, see :cpp:func:`cudf::strings::replace_re`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    patterns: RegexProgram or list[str]
        If RegexProgram, the regex to match to each string.
        If list[str], a list of regex strings to search within each string.

    replacement : Scalar or Column
        If Scalar, the string used to replace the matched sequence in each string.
        ``patterns`` must be a RegexProgram.
        If Column, the strings used for replacement.
        ``patterns`` must be a list[str].

    max_replace_count : int
        The maximum number of times to replace the matched pattern
        within each string. ``patterns`` must be a RegexProgram.
        Default replaces every substring that is matched.

    flags : RegexFlags
        Regex flags for interpreting special characters in the patterns.
        ``patterns`` must be a list[str]

    Returns
    -------
    Column
        New strings column
    """
    cdef unique_ptr[column] c_result
    cdef vector[string] c_lst_patterns
    cdef regex_program* c_regex_pattern
    cdef column_view c_replacement_col
    cdef string_scalar* c_replacement_scalar

    if patterns is RegexProgram and replacement is Scalar:
        c_replacement_scalar = <string_scalar*>((<Scalar>replacement).get())
        c_regex_pattern = (<RegexProgram>patterns).c_obj.get()

        with nogil:
            c_result = move(
                cpp_replace_re.replace_re(
                    input.view(),
                    dereference(c_regex_pattern),
                    dereference(c_replacement_scalar),
                    max_replace_count,
                )
            )

        return Column.from_libcudf(move(c_result))
    elif patterns is list and replacement is Column:
        c_replacement_col = (<Column>replacement).view()
        for pattern in patterns:
            c_lst_patterns.push_back(pattern.encode())

        with nogil:
            c_result = move(
                cpp_replace_re.replace_re(
                    input.view(),
                    c_lst_patterns,
                    c_replacement_col,
                    flags,
                )
            )

        return Column.from_libcudf(move(c_result))
    else:
        raise ValueError("invalid type combinations of patterns and replacement")


cpdef Column replace_with_backrefs(
    Column input,
    RegexProgram prog,
    str replacement
):
    """
    For each string, replaces any character sequence matching the given regex
    using the replacement template for back-references.

    For details, see :cpp:func:`cudf::strings::replace_with_backrefs`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    prog: RegexProgram
        Regex program instance.

    replacement : str
         The replacement template for creating the output string.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    cdef string c_replacement = replacement.encode()

    with nogil:
        c_result = move(
            cpp_replace_re.replace_with_backrefs(
                input.view(),
                prog.c_obj.get()[0],
                c_replacement,
            )
        )

    return Column.from_libcudf(move(c_result))
