from cudf._libxx.lib cimport *
from cudf._libxx.lib import *

from libcpp.pair cimport pair

cdef extern from "cudf/groupby.hpp" \
        namespace "cudf::experimental::groupby" nogil:

    cdef cppclass aggregation_request:
        column_view values
        vector[unique_ptr[aggregation]] aggregations

    cdef cppclass aggregation_result
        vector[unique_ptr[column]] results

    cdef cppclass groups \
            "cudf::experimental::groupby::groupby::groups" nogil:
        unique_ptr[table] group_keys
        vector[size_type] group_offsets
        unique_ptr[table] group_values

    cdef cppclass groupby:
        groupby(table_view keys) except +
        groupby(table_view keys, bool ignore_null_keys) except +

        groupby(
            table_view keys,
            bool ignore_null_keys,
            bool keys_are_sorted,
        ) except +

        groupby(
            table_view keys,
            bool ignore_null_keys,
            bool keys_are_sorted,
            vector[order] column_order,
        ) except +

        groupby(
            table_view keys,
            bool ignore_null_keys,
            bool keys_are_sorted,
            vector[order] column_order,
            vector[null_order] null_precedence
        ) except +

        pair[
            unique_ptr[table],
            vector[aggregation_result]
        ] aggregate(
            vector[aggregation_request] requests,
        ) except +

        groups get_groups() except +
