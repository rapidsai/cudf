# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.aggregation cimport (
    Kind as kind_t,
    aggregation,
    correlation_type,
    groupby_aggregation,
    groupby_scan_aggregation,
    rank_method,
    rank_percentage,
)
from cudf._lib.cpp.types cimport (
    interpolation,
    nan_equality,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
)

from .types cimport DataType


cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj
    cpdef kind(self)
    cdef unique_ptr[groupby_aggregation] clone_underlying_as_groupby(self) except *
    cdef unique_ptr[groupby_scan_aggregation] clone_underlying_as_groupby_scan(
        self
    ) except *

    @staticmethod
    cdef Aggregation from_libcudf(unique_ptr[aggregation] agg)


cpdef Aggregation sum()

cpdef Aggregation product()

cpdef Aggregation min()

cpdef Aggregation max()

cpdef Aggregation count(null_policy null_handling = *)

cpdef Aggregation any()

cpdef Aggregation all()

cpdef Aggregation sum_of_squares()

cpdef Aggregation mean()

cpdef Aggregation variance(size_type ddof = *)

cpdef Aggregation std(size_type ddof = *)

cpdef Aggregation median()

cpdef Aggregation quantile(list quantiles, interpolation interp = *)

cpdef Aggregation argmax()

cpdef Aggregation argmin()

cpdef Aggregation nunique(null_policy null_handling = *)

cpdef Aggregation nth_element(size_type n, null_policy null_handling = *)

cpdef Aggregation collect_list(null_policy null_handling = *)

cpdef Aggregation collect_set(null_handling = *, nulls_equal = *, nans_equal = *)

cpdef Aggregation udf(str operation, DataType output_type)

cpdef Aggregation correlation(correlation_type type, size_type min_periods)

cpdef Aggregation covariance(size_type min_periods, size_type ddof)

cpdef Aggregation rank(
    rank_method method,
    order column_order = *,
    null_policy null_handling = *,
    null_order null_precedence = *,
    rank_percentage percentage = *,
)
