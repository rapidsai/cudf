# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.functional cimport reference_wrapper
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.aggregation cimport (
    groupby_aggregation,
    groupby_scan_aggregation,
)
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.replace cimport replace_policy
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport (
    null_order,
    null_policy,
    order,
    size_type,
    sorted,
)
from cudf._lib.pylibcudf.libcudf.utilities.host_span cimport host_span

# workaround for https://github.com/cython/cython/issues/3885
ctypedef const scalar constscalar


cdef extern from "cudf/groupby.hpp" \
        namespace "cudf::groupby" nogil:

    cdef cppclass aggregation_request:
        aggregation_request() except +
        column_view values
        vector[unique_ptr[groupby_aggregation]] aggregations

    cdef cppclass scan_request:
        scan_request() except +
        column_view values
        vector[unique_ptr[groupby_scan_aggregation]] aggregations

    cdef cppclass aggregation_result:
        vector[unique_ptr[column]] results

    cdef cppclass groups \
            "cudf::groupby::groupby::groups" nogil:
        unique_ptr[table] keys
        vector[size_type] offsets
        unique_ptr[table] values

    cdef cppclass groupby:
        groupby(const table_view& keys) except +
        groupby(
            const table_view& keys,
            null_policy include_null_keys
        ) except +

        groupby(
            const table_view& keys,
            null_policy include_null_keys,
            sorted keys_are_sorted,
        ) except +

        groupby(
            const table_view& keys,
            null_policy include_null_keys,
            sorted keys_are_sorted,
            const vector[order]& column_order,
        ) except +

        groupby(
            const table_view& keys,
            null_policy include_null_keys,
            sorted keys_are_sorted,
            const vector[order]& column_order,
            const vector[null_order]& null_precedence
        ) except +

        pair[
            unique_ptr[table],
            vector[aggregation_result]
        ] aggregate(
            const vector[aggregation_request]& requests,
        ) except +

        pair[
            unique_ptr[table],
            vector[aggregation_result]
        ] scan(
            const vector[scan_request]& requests,
        ) except +

        pair[
            unique_ptr[table],
            unique_ptr[table]
        ] shift(
            const table_view values,
            const vector[size_type] offset,
            const vector[reference_wrapper[constscalar]] fill_values
        ) except +

        groups get_groups() except +
        groups get_groups(table_view values) except +

        pair[unique_ptr[table], unique_ptr[table]] replace_nulls(
            const table_view& values,
            const vector[replace_policy] replace_policy
        ) except +
