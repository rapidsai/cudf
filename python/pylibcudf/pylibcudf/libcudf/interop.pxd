# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pyarrow.lib cimport CScalar, CTable

from cudf._lib.types import cudf_to_np_types, np_to_cudf_types

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "dlpack/dlpack.h" nogil:
    ctypedef struct DLManagedTensor:
        void(*deleter)(DLManagedTensor*) except +


# The Arrow structs are not namespaced.
cdef extern from "cudf/interop.hpp" nogil:
    cdef struct ArrowSchema:
        void (*release)(ArrowSchema*) noexcept nogil

    cdef struct ArrowArray:
        void (*release)(ArrowArray*) noexcept nogil

    cdef struct ArrowArrayStream:
        void (*release)(ArrowArrayStream*) noexcept nogil


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

    cdef shared_ptr[CTable] to_arrow(
        table_view input,
        vector[column_metadata] metadata,
    ) except +

    cdef shared_ptr[CScalar] to_arrow(
        const scalar& input,
        column_metadata metadata,
    ) except +

    cdef unique_ptr[table] from_arrow_stream(ArrowArrayStream* input) except +
    cdef unique_ptr[column] from_arrow_column(
        const ArrowSchema* schema,
        const ArrowArray* input
    ) except +
