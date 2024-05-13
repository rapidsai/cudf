# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.types cimport nan_equality, null_equality


cdef extern from "cudf/lists/stream_compaction.hpp" \
        namespace "cudf::lists" nogil:
    cdef unique_ptr[column] distinct(
        const lists_column_view lists_column,
        null_equality nulls_equal,
        nan_equality nans_equal
    ) except +
