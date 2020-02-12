# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

cdef extern from "cudf/aggregation.hpp" namespace "cudf::experimental" nogil:
    ctypedef enum udf_type:
        CUDA 'cudf::experimental::udf_type::CUDA'
        PTX 'cudf::experimental::udf_type::PTX'

    cdef cppclass aggregation:
        pass

    cdef unique_ptr[aggregation] make_sum_aggregation()

    cdef unique_ptr[aggregation] make_product_aggregation()

    cdef unique_ptr[aggregation] make_min_aggregation()

    cdef unique_ptr[aggregation] make_max_aggregation()

    cdef unique_ptr[aggregation] make_count_aggregation()

    cdef unique_ptr[aggregation] make_any_aggregation()

    cdef unique_ptr[aggregation] make_all_aggregation()

    cdef unique_ptr[aggregation] make_sum_of_squares_aggregation()

    cdef unique_ptr[aggregation] make_mean_aggregation()

    cdef unique_ptr[aggregation] make_variance_aggregation(size_type ddof)

    cdef unique_ptr[aggregation] make_std_aggregation(size_type ddof)

    cdef unique_ptr[aggregation] make_median_aggregation()

    cdef unique_ptr[aggregation] make_quantile_aggregation(vector[double] q,
                                                           interpolation i)

    cdef unique_ptr[aggregation] make_argmax_aggregation()

    cdef unique_ptr[aggregation] make_argmin_aggregation()

    cdef unique_ptr[aggregation] make_udf_aggregation(
        udf_type type,
        string user_defined_aggregator,
        data_type output_type)

cdef unique_ptr[aggregation] get_aggregation(op, kwargs)
