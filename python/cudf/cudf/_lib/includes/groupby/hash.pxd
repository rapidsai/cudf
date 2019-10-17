# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport pair
from libcpp.vector cimport vector

from cudf._lib.cudf import *
from cudf._lib.cudf cimport *

cimport cudf._lib.includes.groupby.common as groupby_common

cdef extern from "cudf/groupby.hpp" namespace "cudf::groupby" nogil:
    ctypedef enum operators:
        SUM,
        MIN,
        MAX,
        COUNT,
        MEAN,
        MEDIAN,
        QUANTILE

cdef extern from "cudf/groupby.hpp" namespace "cudf::groupby::hash" nogil:
    cdef cppclass Options(groupby_common.Options):
        Options(bool _ignore_null_keys) except +

    cdef pair[cudf_table, cudf_table] groupby(
        cudf_table  keys,
        cudf_table values,
        vector[operators] ops,
        Options options
    ) except +
