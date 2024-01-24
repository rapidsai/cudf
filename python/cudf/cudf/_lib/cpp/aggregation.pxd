# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.cpp.types cimport (
    data_type,
    interpolation,
    null_order,
    null_policy,
    order,
    size_type,
)

ctypedef int32_t underlying_type_t_correlation_type
ctypedef int32_t underlying_type_t_rank_method

cdef extern from "cudf/aggregation.hpp" namespace "cudf" nogil:

    # Cython doesn't appear to support enum class nested inside a class, so
    # have to namespace it manually
    cpdef enum class Kind "cudf::aggregation::Kind":
        SUM
        PRODUCT
        MIN
        MAX
        COUNT_VALID
        COUNT_ALL
        ANY
        ALL
        SUM_OF_SQUARES
        MEAN
        VARIANCE
        STD
        MEDIAN
        QUANTILE
        ARGMAX
        ARGMIN
        NUNIQUE
        NTH_ELEMENT
        RANK
        COLLECT_LIST
        COLLECT_SET
        PTX
        CUDA
        CORRELATION
        COVARIANCE

    cdef cppclass aggregation:
        Kind kind

    cdef cppclass rolling_aggregation:
        Kind kind

    cdef cppclass groupby_aggregation:
        Kind kind

    cdef cppclass groupby_scan_aggregation:
        Kind kind

    cdef cppclass reduce_aggregation:
        Kind kind

    cdef cppclass scan_aggregation:
        Kind kind

    cpdef enum class udf_type:
        CUDA
        PTX

    cpdef enum class correlation_type:
        PEARSON
        KENDALL
        SPEARMAN

    cpdef enum class rank_method:
        FIRST
        AVERAGE
        MIN
        MAX
        DENSE

    cpdef enum class rank_percentage:
        NONE
        ZERO_NORMALIZED
        ONE_NORMALIZED

    cdef unique_ptr[T] make_sum_aggregation[T]() except +

    cdef unique_ptr[T] make_product_aggregation[T]() except +

    cdef unique_ptr[T] make_min_aggregation[T]() except +

    cdef unique_ptr[T] make_max_aggregation[T]() except +

    cdef unique_ptr[T] make_count_aggregation[T]() except +

    cdef unique_ptr[T] make_count_aggregation[T](null_policy) except +

    cdef unique_ptr[T] make_any_aggregation[T]() except +

    cdef unique_ptr[T] make_all_aggregation[T]() except +

    cdef unique_ptr[T] make_sum_of_squares_aggregation[T]() except +

    cdef unique_ptr[T] make_mean_aggregation[T]() except +

    cdef unique_ptr[T] make_variance_aggregation[T](
        size_type ddof) except +

    cdef unique_ptr[T] make_std_aggregation[T](size_type ddof) except +

    cdef unique_ptr[T] make_median_aggregation[T]() except +

    cdef unique_ptr[T] make_quantile_aggregation[T](
        vector[double] q, interpolation i) except +

    cdef unique_ptr[T] make_argmax_aggregation[T]() except +

    cdef unique_ptr[T] make_argmin_aggregation[T]() except +

    cdef unique_ptr[T] make_nunique_aggregation[T]() except +

    cdef unique_ptr[T] make_nth_element_aggregation[T](
        size_type n
    ) except +

    cdef unique_ptr[T] make_nth_element_aggregation[T](
        size_type n,
        null_policy null_handling
    ) except +

    cdef unique_ptr[T] make_collect_list_aggregation[T]() except +

    cdef unique_ptr[T] make_collect_set_aggregation[T]() except +

    cdef unique_ptr[T] make_udf_aggregation[T](
        udf_type type,
        string user_defined_aggregator,
        data_type output_type) except +

    cdef unique_ptr[T] make_correlation_aggregation[T](
        correlation_type type, size_type min_periods) except +

    cdef unique_ptr[T] make_covariance_aggregation[T](
        size_type min_periods, size_type ddof) except +

    cdef unique_ptr[T] make_rank_aggregation[T](
        rank_method method,
        order column_order,
        null_policy null_handling,
        null_order null_precedence,
        rank_percentage percentage) except +
