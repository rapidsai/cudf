# Copyright (c) 2023, NVIDIA CORPORATION.


cdef class gpumemoryview:
    """Minimal representation of a memory buffer."""
    def __init__(self, Py_ssize_t ptr):
        self.ptr = ptr
