# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

cdef extern from "dlpack.h" nogil:

    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*)
