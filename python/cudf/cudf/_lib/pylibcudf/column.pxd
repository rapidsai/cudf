# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport offset_type, size_type

from . cimport libcudf_types
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
        # TODO: Is there a more efficient container I could use than a list
        # here? For now defaulting to List[gpumemoryview]
        object children

        # Internals
        libcudf_types.ColumnView _underlying

    cpdef libcudf_types.ColumnView get_underlying(self)
