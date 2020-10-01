# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

from cudf._lib.types import np_to_cudf_types, cudf_to_np_types

from cudf._lib.cpp.types cimport (
    size_type, null_policy, nan_policy, null_equality
)
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cudf/stream_compaction.hpp" namespace "cudf" \
        nogil:
    ctypedef enum duplicate_keep_option:
        KEEP_FIRST 'cudf::duplicate_keep_option::KEEP_FIRST'
        KEEP_LAST 'cudf::duplicate_keep_option::KEEP_LAST'
        KEEP_NONE 'cudf::duplicate_keep_option::KEEP_NONE'

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
                                           null_equality nulls_equal) except +

    cdef size_type distinct_count(column_view source_table,
                                  null_policy null_handling,
                                  nan_policy nan_handling) except +
