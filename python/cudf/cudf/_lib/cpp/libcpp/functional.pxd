# Copyright (c) 2020-2022, NVIDIA CORPORATION.


cdef extern from "<functional>" namespace "std" nogil:
    cdef cppclass reference_wrapper[T]:
        reference_wrapper()
        reference_wrapper(T)
