# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.strings cimport case as cpp_case


cpdef Column to_lower(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.to_lower(input.view())

    return Column.from_libcudf(move(c_result))

cpdef Column to_upper(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.to_upper(input.view())

    return Column.from_libcudf(move(c_result))

cpdef Column swapcase(Column input):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.swapcase(input.view())

    return Column.from_libcudf(move(c_result))
