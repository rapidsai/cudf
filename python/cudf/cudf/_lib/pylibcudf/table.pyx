# Copyright (c) 2023, NVIDIA CORPORATION.


cdef class Table:
    """A set of columns of the same size."""
    def __init__(self, object columns):
        self.column = columns
