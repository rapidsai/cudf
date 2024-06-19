# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from cudf._lib.pylibcudf.libcudf.strings cimport capitalize as cpp_capitalize
from cudf._lib.pylibcudf.scalar cimport Scalar
from cudf._lib.pylibcudf.strings.char_types cimport string_character_types

from cython.operator import dereference


cpdef Column capitalize(
    Column input,
    Scalar delimiters=None
    # TODO: default scalar values
    # https://github.com/rapidsai/cudf/issues/15505
):

    cdef unique_ptr[column] c_result

    if delimiters is None:
        delimiters = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode())
        )

    cdef const string_scalar* cpp_delimiters = <const string_scalar*>(
        delimiters.c_obj.get()
    )

    with nogil:
        c_result = cpp_capitalize.capitalize(
            input.view(),
            dereference(cpp_delimiters)
        )

    return Column.from_libcudf(move(c_result))


cpdef Column title(
    Column input,
    string_character_types sequence_type=string_character_types.ALPHA
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_capitalize.title(input.view(), sequence_type)

    return Column.from_libcudf(move(c_result))


cpdef Column is_title(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_capitalize.is_title(input.view())

    return Column.from_libcudf(move(c_result))
