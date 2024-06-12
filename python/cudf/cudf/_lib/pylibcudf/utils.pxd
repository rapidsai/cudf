# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.types cimport bitmask_type


cdef void * int_to_void_ptr(Py_ssize_t ptr) nogil
cdef bitmask_type * int_to_bitmask_ptr(Py_ssize_t ptr) nogil
cdef vector[reference_wrapper[const scalar]] _as_vector(list source)
