# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool


cdef extern from "<variant>" namespace "std" nogil:
    cdef cppclass variant:
        variant& operator=(variant&)
        size_t index()

    cdef cppclass monostate:
        pass

    cdef T* get_if[T](...)
    cdef bool holds_alternative[T](...)
