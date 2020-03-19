# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types

from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view


cdef extern from "cudf/stream_compaction.hpp" namespace "cudf::experimental" \
        nogil:
    ctypedef enum duplicate_keep_option:
        KEEP_FIRST 'cudf::experimental::duplicate_keep_option::KEEP_FIRST'
        KEEP_LAST 'cudf::experimental::duplicate_keep_option::KEEP_LAST'
        KEEP_NONE 'cudf::experimental::duplicate_keep_option::KEEP_NONE'

    cdef unique_ptr[table] drop_nulls(table_view source_table,
                                      vector[size_type] keys,
                                      size_type keep_threshold) except +

    cdef unique_ptr[table] apply_boolean_mask(
        table_view source_table,
        column_view boolean_mask
    ) except +

    cdef unique_ptr[table] drop_duplicates(table_view source_table,
                                           vector[size_type] keys,
                                           duplicate_keep_option keep,
                                           bool nulls_are_equal) except +

    cdef size_type unique_count(column_view source_table,
                                bool ignore_nulls,
                                bool nan_as_null) except +
