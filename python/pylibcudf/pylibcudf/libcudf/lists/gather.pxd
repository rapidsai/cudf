# Copyright (c) 2021-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.copying cimport out_of_bounds_policy
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view

cdef extern from "cudf/lists/gather.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] segmented_gather(
        const lists_column_view& source_column,
        const lists_column_view& gather_map_list,
        out_of_bounds_policy bounds_policy
    ) except +libcudf_exception_handler
