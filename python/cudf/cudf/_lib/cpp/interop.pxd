# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from pyarrow.lib cimport CTable

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


cdef inline vector[column_metadata] gather_metadata(object metadata) except *:
    """
    Metadata is stored as lists, and expected format is as follows,
    [["a", [["b"], ["c"], ["d"]]],       [["e"]],        ["f", ["", ""]]].
    First value signifies name of the main parent column,
    and adjacent list will signify child column.
    """
    cdef vector[column_metadata] cpp_metadata
    if isinstance(metadata, list):
        cpp_metadata.reserve(len(metadata))
        for i, val in enumerate(metadata):
            cpp_metadata.push_back(column_metadata(str.encode(str(val[0]))))
            if len(val) == 2:
                cpp_metadata[i].children_meta = gather_metadata(val[1])

        return cpp_metadata
    else:
        raise ValueError("Malformed metadata has been encountered")
