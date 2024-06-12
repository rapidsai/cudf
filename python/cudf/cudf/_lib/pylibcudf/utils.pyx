# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cython.operator import dereference

from libc.stdint cimport uintptr_t
from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.types cimport bitmask_type

from .scalar cimport Scalar

# This is a workaround for
# https://github.com/cython/cython/issues/4180
# when creating reference_wrapper[constscalar] in the constructor
ctypedef const scalar constscalar


cdef void * int_to_void_ptr(Py_ssize_t ptr) nogil:
    return <void*><uintptr_t>(ptr)


cdef bitmask_type * int_to_bitmask_ptr(Py_ssize_t ptr) nogil:
    return <bitmask_type*><uintptr_t>(ptr)


cdef vector[reference_wrapper[const scalar]] _as_vector(list source):
    """Make a vector of reference_wrapper[const scalar] from a list of scalars."""
    cdef vector[reference_wrapper[const scalar]] c_scalars
    c_scalars.reserve(len(source))
    cdef Scalar slr
    for slr in source:
        c_scalars.push_back(
            reference_wrapper[constscalar](dereference((<Scalar?>slr).c_obj)))
    return c_scalars
