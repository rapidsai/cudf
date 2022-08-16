# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from pyarrow.lib cimport CTable
from pyarrow.cuda cimport CudaContext, CudaBuffer

from cudf._lib.types import cudf_to_np_types, np_to_cudf_types

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

    cdef cppclass column_metadata:
        column_metadata() except +
        column_metadata(string name_) except +
        string name
        vector[column_metadata] children_meta

    cdef shared_ptr[CTable] to_arrow(
        table_view input,
        vector[column_metadata] metadata,
    ) except +

    cdef vector[char] export_ipc(
        shared_ptr[CudaContext] ctx, table_view input, vector[column_metadata] metadata
    ) except +

    cdef pair[table_view, unique_ptr[vector[shared_ptr[CudaBuffer]]]] import_ipc(
        shared_ptr[CudaContext] ctx, vector[char] ipc_handles
    ) except +
