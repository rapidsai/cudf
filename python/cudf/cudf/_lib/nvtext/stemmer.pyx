# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.move cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.stemmer cimport (
    porter_stemmer_measure as cpp_porter_stemmer_measure
)
from cudf._lib.column cimport Column


def porter_stemmer_measure(Column strings):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_porter_stemmer_measure(c_strings))

    return Column.from_unique_ptr(move(c_result))
