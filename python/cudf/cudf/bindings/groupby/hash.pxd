# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport pair
from libcpp.vector cimport vector

from cudf.bindings.cudf_cpp import *
from cudf.bindings.cudf_cpp cimport *

cimport cudf.bindings.groupby.common as groupby_common

cdef extern from "groupby.hpp" namespace "cudf::groupby::hash" nogil:
    cdef cppclass Options(groupby_common.Options):
        pass

    ctypedef enum operators:
        SUM,
        MIN,
        MAX,
        COUNT,
        MEAN

    cdef pair[cudf_table, cudf_table] groupby(
            cudf_table  keys,
            cudf_table values,
            vector[operators] ops,
            Options options
    ) except +
