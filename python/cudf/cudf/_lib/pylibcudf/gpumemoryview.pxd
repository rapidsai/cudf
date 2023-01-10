# Copyright (c) 2023, NVIDIA CORPORATION.


cdef class gpumemoryview:
    cdef readonly Py_ssize_t ptr
