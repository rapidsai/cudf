# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool

cdef extern from "cudf/wrappers/bool.hpp" namespace "cudf::experimental" nogil:
    ctypedef bool bool8
