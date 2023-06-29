# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport offset_type, size_type

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType


cdef class Column:
    cdef:
        # Core data
        DataType data_type
        size_type size
        gpumemoryview data
        gpumemoryview mask
        size_type null_count
        offset_type offset
        # children: List[Column]
        object children

        # Internals
        unique_ptr[column_view] _underlying

    cdef column_view* get_underlying(self)

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col)
