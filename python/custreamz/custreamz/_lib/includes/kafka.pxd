# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from custreamz._lib.cudf cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cudf/cudf.h" namespace "cudf::io::datasource" nogil:

