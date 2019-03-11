# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

cdef extern from "dlpack.h" nogil:

    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*)
