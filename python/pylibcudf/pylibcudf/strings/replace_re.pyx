# Copyright (c) 2024, NVIDIA CORPORATION.
from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings cimport replace_re as cpp_replace_re
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_flags cimport regex_flags
from pylibcudf.strings.regex_program cimport RegexProgram

__all__ = ["replace_re", "replace_with_backrefs"]

cpdef Column replace_re(
    Column input,
    Patterns patterns,
    Replacement replacement=None,
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
    cdef vector[string] c_patterns

    if Patterns is RegexProgram and Replacement is Scalar:
        if replacement is None:
            replacement = Scalar.from_libcudf(
                cpp_make_string_scalar("".encode())
            )
        with nogil:
            c_result = move(
                cpp_replace_re.replace_re(
                    input.view(),
                    patterns.c_obj.get()[0],
                    dereference(<string_scalar*>(replacement.get())),
                    max_replace_count
                )
            )

        return Column.from_libcudf(move(c_result))
    elif Patterns is list and Replacement is Column:
        c_patterns.reserve(len(patterns))
        for pattern in patterns:
            c_patterns.push_back(pattern.encode())

        with nogil:
            c_result = move(
                cpp_replace_re.replace_re(
                    input.view(),
                    c_patterns,
                    replacement.view(),
                    flags,
                )
            )

        return Column.from_libcudf(move(c_result))
    else:
        raise TypeError("Must pass either a RegexProgram and a Scalar or a list")


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
        c_result = cpp_replace_re.replace_with_backrefs(
            input.view(),
            prog.c_obj.get()[0],
            c_replacement,
        )

    return Column.from_libcudf(move(c_result))
