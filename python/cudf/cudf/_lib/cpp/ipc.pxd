from pyarrow._cuda cimport CCudaContext, CCudaBuffer
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cudf._lib.cpp.interop cimport column_metadata
from cudf._lib.cpp.table.table_view cimport table_view

cdef extern from "cudf/ipc.hpp" namespace "cudf" nogil:
    cdef vector[char] export_ipc(
        shared_ptr[CCudaContext] ctx, table_view input, vector[column_metadata] metadata
    ) except +

    cdef pair[table_view, unique_ptr[vector[shared_ptr[CCudaBuffer]]]] import_ipc(
        shared_ptr[CCudaContext] ctx, vector[char] ipc_handles
    ) except +
