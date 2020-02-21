# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf import *
from cudf._lib.cudf cimport *

cdef extern from "cudf/legacy/groupby.hpp" namespace "cudf::groupby" nogil:
    cdef cppclass Options:
        Options(bool _ignore_null_keys) except +
        Options() except +
        const bool ignore_null_keys
