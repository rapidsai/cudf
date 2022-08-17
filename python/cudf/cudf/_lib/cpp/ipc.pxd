from pyarrow.lib cimport CBuffer
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from cudf._lib.cpp.interop cimport column_metadata
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table

cdef extern from "cudf/ipc.hpp" namespace "cudf" nogil:
    cdef shared_ptr[CBuffer] export_ipc(
        table_view input, vector[column_metadata] metadata
    ) except +

    cdef pair[unique_ptr[table], vector[string]] import_ipc(
        shared_ptr[CBuffer] ipc_handles
    ) except +
