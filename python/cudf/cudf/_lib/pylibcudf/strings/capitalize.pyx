# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from cudf._lib.cpp.strings cimport capitalize as cpp_capitalize
from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.scalar cimport Scalar

from cython.operator import dereference


cpdef Column capitalize(
    Column input,
    Scalar delimiters=Scalar.from_libcudf(
        cpp_make_string_scalar("".encode('utf-8'))
    )
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_capitalize.capitalize(
            input.view(), dereference(delimiters.c_obj)
        )

    return Column.from_libcudf(move(c_result))


cpdef Column title(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_capitalize.title(input.view())

    return Column.from_libcudf(move(c_result))


cpdef Column is_title(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_capitalize.is_title(input.view())

    return Column.from_libcudf(move(c_result))
