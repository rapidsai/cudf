# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport offset_type, size_type

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType


cdef class Column:
    """A container of nullable device data as a column of elements."""
    def __init__(
        self, DataType dtype not None, size_type size, gpumemoryview data,
        gpumemoryview mask, size_type null_count, offset_type offset,
        # TODO: Not sure what the best input is for children, for now just
        # using a List[ColumnView]
        object children
    ):
        self.data_type = dtype
        self.size = size
        self.data = data
        self.mask = mask
        self.null_count = null_count
        self.offset = offset
        self.children = children
