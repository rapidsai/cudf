# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from cudf._lib.pylibcudf.libcudf.strings.replace cimport (
    replace as cpp_replace,
    replace_slice as cpp_replace_slice,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.scalar cimport Scalar
from cudf._lib.pylibcudf.strings.replace cimport ColumnOrScalar


cpdef Column replace(
    Column input,
    ColumnOrScalar target,
    ColumnOrScalar repl,
    size_type maxrepl = -1
):
    """Replaces target string within each string with the specified replacement string.

    Null string entries will return null output string entries.

    For details, see :cpp:func:`replace`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for in each string or Column containing strings
        to search for in the input column.

        If target is a Column, repl must also be a Column.
    repl : Union[Column, Scalar]
        String to replace target with, or Column (of equal length to target)
        of replacement strings.

        If repl is a Column, target must also be a Column.
    maxrepl : size_type, default -1
        Maximum times to replace if target appears multiple times in the input string.
        Default of -1 specifies replace all occurrences of target in each string.

        This option is not supported when target and repl are of type Column.
        (all occurrences of target will always be replaced in that case)

    Returns
    -------
    pylibcudf.Column
        New string column with target replaced.
    """
    cdef:
        unique_ptr[column] c_result
        const string_scalar* target_str
        const string_scalar* repl_str

    if ColumnOrScalar is Scalar:
        target_str = <string_scalar *>(target.c_obj.get())
        repl_str = <string_scalar *>(repl.c_obj.get())

        with nogil:
            c_result = move(cpp_replace(
                input.view(),
                target_str[0],
                repl_str[0],
                maxrepl,
            ))
    else:
        # Column case

        if maxrepl != -1:
            raise ValueError("maxrepl is not supported as a valid "
                             "argument when target and repl are Columns")

        with nogil:
            c_result = move(cpp_replace(
                input.view(),
                target.view(),
                repl.view(),
            ))

    return Column.from_libcudf(move(c_result))


cpdef Column replace_slice(
    Column input,
    # TODO: default scalar values
    # https://github.com/rapidsai/cudf/issues/15505
    Scalar repl = None,
    size_type start = 0,
    size_type stop = -1
):
    """Replaces each string in the column with the provided repl string
    within the [start,stop) character position range.

    Null string entries will return null output string entries.
    This function can be used to insert a string into specific position
    by specifying the same position value for start and stop.
    The repl string can be appended to each string by specifying -1
    for both start and stop.

    For details, see :cpp:func:`replace_slice`.

    Parameters
    ----------
    input : Column
        The input strings
    repl : Scalar, default ""
        String scalar to replace target with.
    start : size_type, default 0
        Start position where repl will be added.
    stop : size_type, default -1
        End position (exclusive) to use for replacement.
    Returns
    -------
    pylibcudf.Column
        New string column
    """
    cdef unique_ptr[column] c_result

    if repl is None:
        repl = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode())
        )

    cdef const string_scalar* scalar_str = <string_scalar*>(repl.c_obj.get())

    with nogil:
        c_result = move(cpp_replace_slice(
            input.view(),
            scalar_str[0],
            start,
            stop
        ))

    return Column.from_libcudf(move(c_result))
