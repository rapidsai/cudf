# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)


cdef extern from "cudf/lists/gather.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] segmented_gather(
        const lists_column_view source_column,
        const lists_column_view gather_map_list
    ) except +
