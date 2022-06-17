# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.dictionary.dictionary_column_view cimport (
    dictionary_column_view,
)


cdef extern from "cudf/dictionary/update_keys.hpp" \
        namespace "cudf::dictionary" nogil:
    cdef unique_ptr[column] add_keys(
        const dictionary_column_view dictionary_column,
        const column_view new_keys
    )

    cdef unique_ptr[column] remove_keys(
        const dictionary_column_view dictionary_column,
        const column_view keys_to_remove
    )

    cdef unique_ptr[column] remove_unused_keys(
        const dictionary_column_view dictionary_column
    )

    cdef unique_ptr[column] set_keys(
        const dictionary_column_view dictionary_column,
        const column_view keys
    )
