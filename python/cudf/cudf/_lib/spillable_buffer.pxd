# Copyright (c) 2022, NVIDIA CORPORATION.


from libc.stdint cimport uintptr_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

ctypedef vector[shared_ptr[void]] OwnersVecT  # Type aliasing

cdef class SpillableBuffer:
    cdef object __weakref__
    cdef shared_ptr[void] _expose_counter
    cdef object _lock
    cdef bint _exposed
    cdef double _last_accessed
    cdef object _view_desc
    cdef object _ptr_desc
    cdef readonly uintptr_t _ptr
    cdef size_t _size
    cdef object _owner
    cdef object _manager

    cdef void* ptr_raw(self, shared_ptr[OwnersVecT] owners) except *
