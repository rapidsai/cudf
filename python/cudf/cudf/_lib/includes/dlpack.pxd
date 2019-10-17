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

from cudf._lib.cudf cimport *


cdef extern from "dlpack/dlpack.h" nogil:

    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*)


cdef extern from "cudf.h" nogil:

    cdef gdf_error gdf_from_dlpack(
        gdf_column** columns,
        size_type* num_columns,
        const DLManagedTensor* tensor
    ) except +

    cdef gdf_error gdf_to_dlpack(
        DLManagedTensor* tensor,
        const gdf_column** columns,
        size_type num_columns
    ) except +
