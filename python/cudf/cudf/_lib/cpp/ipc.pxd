from pyarrow.cuda cimport CudaContext, CudaBuffer
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector
from cudf._lib.cpp.interop import column_metadata
from cudf._lib.cpp.table.table_view cimport table_view

cdef extern from "cudf/ipc.hpp" namespace "cudf" nogil:
    cdef vector[char] export_ipc(
        shared_ptr[CudaContext] ctx, table_view input, vector[column_metadata] metadata
    ) except +

    cdef pair[table_view, unique_ptr[vector[shared_ptr[CudaBuffer]]]] import_ipc(
        shared_ptr[CudaContext] ctx, vector[char] ipc_handles
    ) except +
