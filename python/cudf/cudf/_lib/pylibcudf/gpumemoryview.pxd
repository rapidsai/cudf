# Copyright (c) 2023, NVIDIA CORPORATION.


cdef class gpumemoryview:
    # TODO: Probably should use ssize_t
    # TODO: Is an int sufficient, or do we need something larger?
    cdef readonly Py_ssize_t ptr
    cdef readonly Py_ssize_t size
