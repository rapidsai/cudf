# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string

from pyarrow.lib cimport CTable
from cudf._lib.types import np_to_cudf_types, cudf_to_np_types

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view

cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*) except +

cdef extern from "cudf/interop.hpp" namespace "cudf" \
        nogil:
    cdef unique_ptr[table] from_dlpack(const DLManagedTensor* tensor
                                       ) except +

    DLManagedTensor* to_dlpack(table_view input_table
                               ) except +

    cdef unique_ptr[table] from_arrow(CTable input) except +

    cdef shared_ptr[CTable] to_arrow(
        table_view input,
        vector[string] column_names
    ) except +
