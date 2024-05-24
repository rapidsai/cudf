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
