# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.types cimport (
    size_type,
    data_type,
    interpolation,
    null_policy
)


cdef extern from "cudf/aggregation.hpp" namespace "cudf::experimental" nogil:

    cdef cppclass aggregation:
        ctypedef enum Kind:
            SUM 'cudf::experimental::aggregation::SUM'
            PRODUCT 'cudf::experimental::aggregation::PRODUCT'
            MIN 'cudf::experimental::aggregation::MIN'
            MAX 'cudf::experimental::aggregation::MAX'
            COUNT_VALID 'cudf::experimental::aggregation::COUNT_VALID'
            COUNT_ALL 'cudf::experimental::aggregation::COUNT_ALL'
            ANY 'cudf::experimental::aggregation::ANY'
            ALL 'cudf::experimental::aggregation::ALL'
            SUM_OF_SQUARES 'cudf::experimental::aggregation::SUM_OF_SQUARES'
            MEAN 'cudf::experimental::aggregation::MEAN'
            VARIANCE 'cudf::experimental::aggregation::VARIANCE'
            STD 'cudf::experimental::aggregation::STD'
            MEDIAN 'cudf::experimental::aggregation::MEDIAN'
            QUANTILE 'cudf::experimental::aggregation::QUANTILE'
            ARGMAX 'cudf::experimental::aggregation::ARGMAX'
            ARGMIN 'cudf::experimental::aggregation::ARGMIN'
            NUNIQUE 'cudf::experimental::aggregation::NUNIQUE'
            NTH_ELEMENT 'cudf::experimental::aggregation::NTH_ELEMENT'
            PTX 'cudf::experimental::aggregation::PTX'
            CUDA 'cudf::experimental::aggregation::CUDA'
        Kind kind

    ctypedef enum udf_type:
        CUDA 'cudf::experimental::udf_type::CUDA'
        PTX 'cudf::experimental::udf_type::PTX'

    cdef unique_ptr[aggregation] make_sum_aggregation() except +

    cdef unique_ptr[aggregation] make_product_aggregation() except +

    cdef unique_ptr[aggregation] make_min_aggregation() except +

    cdef unique_ptr[aggregation] make_max_aggregation() except +

    cdef unique_ptr[aggregation] make_count_aggregation() except +

    cdef unique_ptr[aggregation] make_count_aggregation(null_policy) except +

    cdef unique_ptr[aggregation] make_any_aggregation() except +

    cdef unique_ptr[aggregation] make_all_aggregation() except +

    cdef unique_ptr[aggregation] make_sum_of_squares_aggregation() except +

    cdef unique_ptr[aggregation] make_mean_aggregation() except +

    cdef unique_ptr[aggregation] make_variance_aggregation(
        size_type ddof) except +

    cdef unique_ptr[aggregation] make_std_aggregation(size_type ddof) except +

    cdef unique_ptr[aggregation] make_median_aggregation() except +

    cdef unique_ptr[aggregation] make_quantile_aggregation(
        vector[double] q, interpolation i) except +

    cdef unique_ptr[aggregation] make_argmax_aggregation() except +

    cdef unique_ptr[aggregation] make_argmin_aggregation() except +

    cdef unique_ptr[aggregation] make_nunique_aggregation() except +

    cdef unique_ptr[aggregation] make_nth_element_aggregation(
        size_type n
    ) except +

    cdef unique_ptr[aggregation] make_udf_aggregation(
        udf_type type,
        string user_defined_aggregator,
        data_type output_type) except +
