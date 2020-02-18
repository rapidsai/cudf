# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr


cpdef commit_offsets():
    print("Committing offsets!!!")
