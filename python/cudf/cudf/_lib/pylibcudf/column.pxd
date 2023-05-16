# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport offset_type, size_type

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType


cdef class Column:
    cdef DataType data_type
    cdef size_type size
    cdef gpumemoryview data
    cdef gpumemoryview mask
    cdef size_type null_count
    cdef offset_type offset
    # TODO: Is there a more efficient container I could use than a list here?
    # For now defaulting to List[gpumemoryview]
    cdef object children
