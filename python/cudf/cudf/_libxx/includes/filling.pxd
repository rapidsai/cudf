# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *


cdef extern from "cudf/filling.hpp" namespace "cudf::experimental" nogil:
    cdef void fill_in_place(
        mutable_column_view destination,
        size_type beign,
        size_type end,
        scalar value
    ) except +


    cdef unique_ptr[column] fill(
        column_view input,
        size_type begin,
        size_type end,
        scalar value
    ) except +


    cdef unique_ptr[table] repeat(
        table_view input,
        column_view count,
        bool check_count
    ) except +


    cdef unique_ptr[table] repeat(
        table_view input,
        scalar count
    ) except +
