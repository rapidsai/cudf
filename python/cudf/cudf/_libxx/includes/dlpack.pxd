# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport unique_ptr

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types

from cudf._libxx.includes.table.table cimport table
from cudf._libxx.includes.table.table_view cimport table_view

cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*) except +

cdef extern from "cudf/dlpack.hpp" namespace "cudf" \
        nogil:
    cdef unique_ptr[table] from_dlpack(const DLManagedTensor* tensor
                                       ) except +

    DLManagedTensor* to_dlpack(table_view input_table
                               ) except +
