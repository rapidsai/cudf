# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr


cdef extern from "<memory>" namespace "std" nogil:
    # The Cython standard header does not have except +, so C++
    # exceptions from make_unique are not caught and translated to
    # Python ones. This is not perfectly ergonomic, we always have to
    # wrap make_unique in move, but at least we can catch exceptions.
    # See https://github.com/cython/cython/issues/5560
    unique_ptr[T] make_unique[T](...) except +
