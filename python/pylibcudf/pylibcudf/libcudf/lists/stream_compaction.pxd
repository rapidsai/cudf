# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport nan_equality, null_equality


cdef extern from "cudf/lists/stream_compaction.hpp" \
        namespace "cudf::lists" nogil:
    cdef unique_ptr[column] apply_boolean_mask(
        const lists_column_view& lists_column,
        const lists_column_view& boolean_mask,
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] distinct(
        const lists_column_view& lists_column,
        null_equality nulls_equal,
        nan_equality nans_equal
    ) except +libcudf_exception_handler
