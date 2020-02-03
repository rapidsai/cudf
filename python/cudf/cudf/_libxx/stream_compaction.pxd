# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

cdef extern from "cudf/stream_compaction.hpp" namespace "cudf::experimental" \
        nogil:
    ctypedef enum duplicate_keep_option:
        KEEP_FIRST 'cudf::experimental::duplicate_keep_option::KEEP_FIRST'
        KEEP_LAST 'cudf::experimental::duplicate_keep_option::KEEP_LAST'
        KEEP_NONE 'cudf::experimental::duplicate_keep_option::KEEP_NONE'

    cdef unique_ptr[table] drop_nulls(table_view source_table,
                                      vector[size_type] keys,
                                      size_type keep_threshold)

    cdef unique_ptr[table] apply_boolean_mask(table_view source_table,
                                              column_view boolean_mask)

    cdef unique_ptr[table] drop_duplicates(table_view source_table,
                                           vector[size_type] keys,
                                           duplicate_keep_option keep,
                                           bool nulls_are_equal)

    cdef size_type unique_count(column_view source_table,
                                bool ignore_nulls,
                                bool nan_as_null)
