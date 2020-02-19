# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

cdef extern from "cudf/aggregation.hpp" namespace "cudf::experimental" nogil:
    ctypedef enum udf_type:
        CUDA 'cudf::experimental::udf_type::CUDA'
        PTX 'cudf::experimental::udf_type::PTX'

    cdef unique_ptr[aggregation] make_sum_aggregation() except +

    cdef unique_ptr[aggregation] make_product_aggregation() except +

    cdef unique_ptr[aggregation] make_min_aggregation() except +

    cdef unique_ptr[aggregation] make_max_aggregation() except +

    cdef unique_ptr[aggregation] make_count_aggregation() except +

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

    cdef unique_ptr[aggregation] make_udf_aggregation(
        udf_type type,
        string user_defined_aggregator,
        data_type output_type) except +

cdef unique_ptr[aggregation] get_aggregation(op, kwargs) except *
