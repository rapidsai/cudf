# Copyright (c) 2023, NVIDIA CORPORATION.


cdef class gpumemoryview:
    cdef Py_ssize_t ptr
    cdef object obj
