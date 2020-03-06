from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp cimport bool

from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.aggregation cimport aggregation
from cudf._libxx.cpp.types cimport size_type, order, null_order, include_nulls


cdef extern from "cudf/groupby.hpp" \
        namespace "cudf::experimental::groupby" nogil:

    cdef cppclass aggregation_request:
        aggregation_request() except +
        column_view values
        vector[unique_ptr[aggregation]] aggregations

    cdef cppclass aggregation_result:
        vector[unique_ptr[column]] results

    cdef cppclass groups \
            "cudf::experimental::groupby::groupby::groups" nogil:
        unique_ptr[table] keys
        vector[size_type] offsets
        unique_ptr[table] values

    cdef cppclass groupby:
        groupby(const table_view& keys) except +
        groupby(const table_view& keys, include_nulls include_null_keys) except +

        groupby(
            table_view keys,
            include_nulls include_null_keys,
            bool keys_are_sorted,
        ) except +

        groupby(
            table_view keys,
            include_nulls include_null_keys,
            bool keys_are_sorted,
            vector[order] column_order,
        ) except +

        groupby(
            table_view keys,
            include_nulls include_null_keys,
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
        groups get_groups(table_view values) except +
