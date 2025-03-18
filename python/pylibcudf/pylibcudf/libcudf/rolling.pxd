# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.aggregation cimport rolling_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_order, order, size_type


cdef extern from "cudf/rolling.hpp" namespace "cudf" nogil:
    cdef cppclass rolling_request:
        rolling_request() except +libcudf_exception_handler
        column_view values
        unique_ptr[rolling_aggregation] aggregation
    cdef cppclass bounded_closed:
        bounded_closed(const scalar&) noexcept
    cdef cppclass bounded_open:
        bounded_open(const scalar&) noexcept
    cdef cppclass unbounded:
        unbounded() noexcept
    cdef cppclass current_row:
        current_row() noexcept

    # In the C++ API, there's just a single function that takes a std::variant
    # But we have no way of passing these from Cython so lie in the overloads here.
    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_closed preceding,
        bounded_closed following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_open preceding,
        bounded_open following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        current_row preceding,
        current_row following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        unbounded preceding,
        unbounded following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_closed preceding,
        bounded_open following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_closed preceding,
        current_row following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_closed preceding,
        unbounded following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_open preceding,
        bounded_closed following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_open preceding,
        current_row following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        bounded_open preceding,
        unbounded following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        current_row preceding,
        bounded_closed following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        current_row preceding,
        bounded_open following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        current_row preceding,
        unbounded following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        unbounded preceding,
        bounded_closed following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        unbounded preceding,
        bounded_open following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        unbounded preceding,
        current_row following,
        size_type min_periods,
        vector[rolling_request]& requests
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] rolling_window(
        column_view source,
        column_view preceding_window,
        column_view following_window,
        size_type min_periods,
        rolling_aggregation& agg) except +libcudf_exception_handler

    cdef unique_ptr[column] rolling_window(
        column_view source,
        size_type preceding_window,
        size_type following_window,
        size_type min_periods,
        rolling_aggregation& agg) except +libcudf_exception_handler
