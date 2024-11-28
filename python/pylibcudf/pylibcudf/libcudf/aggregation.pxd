# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libc.stddef cimport size_t
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport (
    data_type,
    interpolation,
    nan_equality,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
)


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
        EWMA

    cdef cppclass aggregation:
        Kind kind
        unique_ptr[aggregation] clone()
        size_t do_hash() noexcept
        bool is_equal(const aggregation const) noexcept

    cdef cppclass rolling_aggregation(aggregation):
        pass

    cdef cppclass groupby_aggregation(aggregation):
        pass

    cdef cppclass groupby_scan_aggregation(aggregation):
        pass

    cdef cppclass reduce_aggregation(aggregation):
        pass

    cdef cppclass scan_aggregation(aggregation):
        pass

    cdef cppclass segmented_reduce_aggregation(aggregation):
        pass

    cpdef enum class udf_type(bool):
        CUDA
        PTX

    cpdef enum class correlation_type(int32_t):
        PEARSON
        KENDALL
        SPEARMAN

    cpdef enum class ewm_history(int32_t):
        INFINITE
        FINITE

    cpdef enum class rank_method(int32_t):
        FIRST
        AVERAGE
        MIN
        MAX
        DENSE

    cpdef enum class rank_percentage(int32_t):
        NONE
        ZERO_NORMALIZED
        ONE_NORMALIZED

    cdef unique_ptr[T] make_sum_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_product_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_min_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_max_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_count_aggregation[T](
        null_policy
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_any_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_all_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_sum_of_squares_aggregation[T]()\
        except +libcudf_exception_handler

    cdef unique_ptr[T] make_mean_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_variance_aggregation[T](
        size_type ddof) except +libcudf_exception_handler

    cdef unique_ptr[T] make_std_aggregation[T](
        size_type ddof
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_median_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_quantile_aggregation[T](
        vector[double] q, interpolation i) except +libcudf_exception_handler

    cdef unique_ptr[T] make_argmax_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_argmin_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_nunique_aggregation[T](
        null_policy null_handling
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_nth_element_aggregation[T](
        size_type n,
        null_policy null_handling
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_collect_list_aggregation[T](
        null_policy null_handling
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_collect_set_aggregation[T](
        null_policy null_handling, null_equality nulls_equal, nan_equality nans_equal
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_udf_aggregation[T](
        udf_type type,
        string user_defined_aggregator,
        data_type output_type) except +libcudf_exception_handler

    cdef unique_ptr[T] make_ewma_aggregation[T](
        double com, ewm_history adjust
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_correlation_aggregation[T](
        correlation_type type, size_type min_periods) except +libcudf_exception_handler

    cdef unique_ptr[T] make_covariance_aggregation[T](
        size_type min_periods, size_type ddof) except +libcudf_exception_handler

    cdef unique_ptr[T] make_rank_aggregation[T](
        rank_method method,
        order column_order,
        null_policy null_handling,
        null_order null_precedence,
        rank_percentage percentage) except +libcudf_exception_handler

cdef extern from "cudf/detail/aggregation/aggregation.hpp" \
        namespace "cudf::detail" nogil:

    cdef cppclass std_var_aggregation(
        rolling_aggregation,
        groupby_aggregation,
        reduce_aggregation,
        segmented_reduce_aggregation
    ):
        size_type _ddof

    cdef cppclass quantile_aggregation(groupby_aggregation, reduce_aggregation):
        vector[double] _quantiles
        interpolation _interpolation

    cdef cppclass nunique_aggregation(
        groupby_aggregation, reduce_aggregation, segmented_reduce_aggregation
    ):
        null_policy _null_handling

    cdef cppclass nth_element_aggregation(
        groupby_aggregation, reduce_aggregation, rolling_aggregation
    ):
        size_type _n
        null_policy _null_handling

    cdef cppclass ewma_aggregation(scan_aggregation):
        double center_of_mass
        ewm_history history

    cdef cppclass rank_aggregation(
        rolling_aggregation, groupby_scan_aggregation, reduce_aggregation
    ):
        rank_method _method
        order _column_order
        null_policy _null_handling
        null_order _null_precedence
        rank_percentage _percentage

    cdef cppclass collect_list_aggregation(
        rolling_aggregation, groupby_aggregation, reduce_aggregation
    ):
        null_policy _null_handling

    cdef cppclass collect_set_aggregation(
        rolling_aggregation, groupby_aggregation, reduce_aggregation
    ):
        null_policy _null_handling
        null_equality _nulls_equal
        nan_equality _nans_equal

    cdef cppclass udf_aggregation(rolling_aggregation):
        string _source
        data_type _output_type

    cdef cppclass correlation_aggregation(groupby_aggregation):
        correlation_type _type
        size_type _min_periods

    cdef cppclass covariance_aggregation(groupby_aggregation):
        size_type _min_periods
        size_type _ddof
