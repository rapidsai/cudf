# Copyright (c) 2020-2023, NVIDIA CORPORATION.


# TODO: Can be replaced once https://github.com/cython/cython/pull/5671 is
# merged and released
cdef extern from "<functional>" namespace "std" nogil:
    cdef cppclass reference_wrapper[T]:
        reference_wrapper()
        reference_wrapper(T)
