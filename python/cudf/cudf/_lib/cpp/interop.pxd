# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pyarrow.lib cimport CScalar, CTable

from cudf._lib.types import cudf_to_np_types, np_to_cudf_types

from cudf._lib.cpp.scalar.scalar cimport scalar
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
    cdef unique_ptr[scalar] from_arrow(CScalar input) except +

    cdef cppclass column_metadata:
        column_metadata() except +
        column_metadata(string name_) except +
        string name
        vector[column_metadata] children_meta

    # TODO: Adding this for pylibcudf because in pylibcudf we don't have column
    # names. However we need to figure out how this will propagate up when cudf
    # starts using pylibcudf for interop functionality.
    cdef shared_ptr[CTable] to_arrow(
        table_view input,
    ) except +

    cdef shared_ptr[CTable] to_arrow(
        table_view input,
        vector[column_metadata] metadata,
    ) except +

    cdef shared_ptr[CScalar] to_arrow(
        const scalar& input,
    ) except +
