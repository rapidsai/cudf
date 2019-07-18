# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp import *
from cudf.bindings.cudf_cpp cimport *

cdef extern from "groupby.hpp" namespace "cudf::groupby":
    cdef cppclass Options:
        Options(bool _ignore_null_keys) except +
        Options() except +
        const bool ignore_null_keys
