# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport bitmask_type
from rmm.pylibrmm.stream cimport Stream


cdef void * int_to_void_ptr(Py_ssize_t ptr) nogil
cdef bitmask_type * int_to_bitmask_ptr(Py_ssize_t ptr) nogil
cdef vector[reference_wrapper[const scalar]] _as_vector(list source)
cdef Stream _get_stream(Stream stream = *)
