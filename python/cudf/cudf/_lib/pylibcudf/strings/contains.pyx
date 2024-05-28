# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.strings cimport contains as cpp_contains
from cudf._lib.pylibcudf.strings.regex_program cimport RegexProgram
from cudf._lib.pylibcudf.scalar cimport Scalar

from cython.operator import dereference

from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar

cpdef Column contains_re(
    Column input,
    RegexProgram prog
):
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_contains.contains_re(
            input.view(),
            prog.c_obj.get()[0]
        )

    return Column.from_libcudf(move(result))

