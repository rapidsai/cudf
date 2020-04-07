# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint8_t
from rmm._lib.device_buffer cimport device_buffer
from cudf._lib.cpp.types cimport (
    size_type,
)
from cudf._lib.cpp.aggregation cimport aggregation
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.column.column cimport column, column_contents
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.groupby cimport (
    groups,
    aggregation_request,
    aggregation_result
)
from cudf._lib.cpp.table.table_view cimport table_view
from pyarrow.includes.libarrow cimport CMessageReader
cimport cudf._lib.cpp.io.types as cudf_io_types

cdef extern from * namespace "cython_std" nogil:
    """
    #if __cplusplus > 199711L
    #include <type_traits>
    namespace cython_std {
    template <typename T> typename std::remove_reference<T>::type&&
    move(T& t) noexcept { return std::move(t); }
    template <typename T> typename std::remove_reference<T>::type&&
    move(T&& t) noexcept { return std::move(t); }
    }
    #endif
    """
    cdef T move[T](T)
