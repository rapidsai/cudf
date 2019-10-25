# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf import *
from cudf._lib.cudf cimport *

cdef extern from "cudf/legacy/groupby.hpp" namespace "cudf::groupby" nogil:
    cdef cppclass Options:
        Options(bool _ignore_null_keys) except +
        Options() except +
        const bool ignore_null_keys
