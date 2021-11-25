# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cudf/dictionary/dictionary_column_view.hpp" \
        namespace "cudf" nogil:
    cdef cppclass dictionary_column_view:
        dictionary_column_view(const column_view dictionary_column)
